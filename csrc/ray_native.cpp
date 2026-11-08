// Ray evidence kernels and an occupancy membership index. No pose or LIO state.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using I = std::int64_t;
using Key = std::array<I, 3>;
using Floats = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Ints = py::array_t<I, py::array::c_style | py::array::forcecast>;
struct Hash {
  std::size_t operator()(const Key& k) const {
    std::size_t h = 0;
    for (I x : k) h ^= std::hash<I>{}(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};
using Keys = std::unordered_set<Key, Hash>;
using Group = std::array<I, 5>;
struct GroupHash {
  std::size_t operator()(const Group& k) const {
    std::size_t h=0;
    for (I x : k) h ^= std::hash<I>{}(x) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
    return h;
  }
};

py::array_t<I> select_grouped(Ints groups,Floats distance) {
  if (groups.ndim()!=2 || groups.shape(1)!=5 || distance.ndim()!=1 || distance.shape(0)!=groups.shape(0))
    throw std::invalid_argument("invalid ray selection arguments");
  auto g=groups.unchecked<2>(); auto d=distance.unchecked<1>();
  std::vector<std::pair<Group,I>> selected;
  {
    py::gil_scoped_release release;
    std::unordered_map<Group,I,GroupHash> nearest;
    nearest.reserve(g.shape(0));
    for (py::ssize_t i=0;i<g.shape(0);++i) {
      if (!std::isfinite(d(i))) throw std::invalid_argument("nonfinite ray distance");
      Group key{g(i,0),g(i,1),g(i,2),g(i,3),g(i,4)};
      auto inserted=nearest.emplace(key,i);
      // Stable ties match lexsort: the earliest input wins equal distances.
      if (!inserted.second && d(i)<d(inserted.first->second)) inserted.first->second=i;
    }
    selected.assign(nearest.begin(),nearest.end());
    std::sort(selected.begin(),selected.end());
  }
  py::array_t<I> result(selected.size());
  for (std::size_t i=0;i<selected.size();++i) result.mutable_data()[i]=selected[i].second;
  return result;
}

I floor_index(double x) {
  // Leave room for +/- 1 neighbour queries. Avoid undefined float->int casts.
  constexpr double bound = 9.0e18;
  if (!std::isfinite(x) || x <= -bound || x >= bound)
    throw std::invalid_argument("ray coordinate exceeds supported integer range");
  return static_cast<I>(std::floor(x));
}
Key read_key(py::handle value) {
  auto key = py::cast<py::tuple>(value);
  if (key.size() != 3) throw std::invalid_argument("voxel key must have three coordinates");
  Key result{py::cast<I>(key[0]), py::cast<I>(key[1]), py::cast<I>(key[2])};
  for (I v : result) if (v <= -9000000000000000000LL || v >= 9000000000000000000LL)
    throw std::invalid_argument("voxel key exceeds supported integer range");
  return result;
}
py::array_t<I> key_array(const std::vector<Key>& keys) {
  py::array_t<I> result({static_cast<py::ssize_t>(keys.size()), py::ssize_t(3)});
  auto out = result.mutable_unchecked<2>();
  for (std::size_t i=0; i<keys.size(); ++i)
    for (int j=0; j<3; ++j) out(i,j)=keys[i][j];
  return result;
}

struct VoxelIndex {
  Keys cells;
  std::mutex mutex;
  void change(Ints keys,bool insert) {
    if (keys.ndim()!=2 || keys.shape(1)!=3) throw std::invalid_argument("voxel keys must be (N,3)");
    auto data=keys.unchecked<2>();
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex);
    for (py::ssize_t i=0;i<data.shape(0);++i) {
      Key key{data(i,0),data(i,1),data(i,2)};
      for (I v : key) if (v<=-9000000000000000000LL || v>=9000000000000000000LL)
        throw std::invalid_argument("voxel key exceeds supported integer range");
      if (insert) cells.insert(key); else cells.erase(key);
    }
  }
  void clear() {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex);
    cells.clear();
  }
};

py::tuple voxel_evidence(Floats rays, Floats selected, Floats lengths,
                         VoxelIndex& index, py::dict misses, double resolution, double max_range) {
  if (rays.ndim()!=2 || rays.shape(1)!=6 || selected.ndim()!=2 || selected.shape(1)!=6 ||
      lengths.ndim()!=1 || lengths.shape(0)!=selected.shape(0) ||
      !std::isfinite(resolution) || resolution<=0 || !std::isfinite(max_range) || max_range<=0)
    throw std::invalid_argument("invalid voxel ray kernel arguments");
  std::vector<Key> miss_keys, reset, crossed;
  for (auto item : misses) miss_keys.push_back(read_key(item.first));
  auto r=rays.unchecked<2>(); auto s=selected.unchecked<2>(); auto len=lengths.unchecked<1>();
  {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(index.mutex);
    const double inv=1./resolution;
    Keys hits, visited;
    hits.reserve(r.shape(0));
    for (py::ssize_t i=0;i<r.shape(0);++i)
      hits.insert({floor_index(r(i,3)*inv),floor_index(r(i,4)*inv),floor_index(r(i,5)*inv)});
    auto protected_key = [&](const Key& key) {
      for (I x=-1;x<=1;++x) for (I y=-1;y<=1;++y) for (I z=-1;z<=1;++z)
        if (hits.count({key[0]+x,key[1]+y,key[2]+z})) return true;
      return false;
    };
    for (const auto& key : miss_keys) if (protected_key(key)) reset.push_back(key);
    // Match numpy.arange(start, stop, step): its actual increment is
    // (start + step) - start. Preserve the Python arithmetic order; no FMA.
    const double step=resolution*.5, increment=(resolution+step)-resolution;
    if (increment<=0) throw std::invalid_argument("voxel resolution is too small");
    for (py::ssize_t i=0;i<s.shape(0);++i) {
      if (!std::isfinite(len(i)) || len(i)<=0) throw std::invalid_argument("invalid ray length");
      for (int j=0;j<6;++j) if (!std::isfinite(s(i,j))) throw std::invalid_argument("nonfinite ray");
      const double limit=std::min(len(i)-2*resolution,max_range);
      double batch_limit=0.;
      const py::ssize_t batch_start=(i/128)*128;
      for (py::ssize_t b=batch_start;b<std::min(batch_start+128,s.shape(0));++b)
        batch_limit=std::max(batch_limit,std::min(len(b)-2*resolution,max_range));
      const double count=std::ceil((batch_limit-resolution)/step);
      if (!std::isfinite(count) || count>10000000.) throw std::invalid_argument("ray sampling limit exceeded");
      for (I n=0;n<static_cast<I>(std::max(0.,count));++n) {
        const double travel=resolution+static_cast<double>(n)*increment;
        if (!(travel<limit)) break;
        const double fraction=travel/len(i);
        Key key; bool interior=true;
        for (int j=0;j<3;++j) {
          const double scaled=(s(i,j)+(s(i,j+3)-s(i,j))*fraction)*inv;
          key[j]=floor_index(scaled);
          const double offset=scaled-static_cast<double>(key[j]);
          interior = interior && offset>.2 && offset<.8;
        }
        if (interior && index.cells.count(key)) visited.insert(key);
      }
    }
    for (const auto& key : visited) if (!protected_key(key)) crossed.push_back(key);
    std::sort(reset.begin(),reset.end()); std::sort(crossed.begin(),crossed.end());
  }
  return py::make_tuple(key_array(reset),key_array(crossed));
}

I floor_div(I a,I b) { I q=a/b; return q-((a%b)<0); }
py::array_t<I> grid_free_keys(Ints rc, Floats z, double min_height, double max_height,
                              double height_resolution,I depth,I width,I height) {
  if (rc.ndim()!=2 || rc.shape(1)!=4 || z.ndim()!=2 || z.shape(1)!=2 || z.shape(0)!=rc.shape(0) ||
      !std::isfinite(min_height) || !std::isfinite(max_height) || max_height<min_height ||
      !std::isfinite(height_resolution) || height_resolution<=0 || depth<=0 || width<=0 || height<=0 ||
      width>std::numeric_limits<I>::max()/height/depth)
    throw std::invalid_argument("invalid grid ray kernel arguments");
  auto coords=rc.unchecked<2>(); auto heights=z.unchecked<2>();
  std::vector<I> keys;
  {
    py::gil_scoped_release release;
    // A bit per existing score avoids hash-node allocation and sorting every
    // repeated beam sample. Memory is 1/8 of the already allocated score grid.
    const I count=width*height*depth;
    std::vector<std::uint64_t> seen(static_cast<std::size_t>(count/64+1),0);
    for (py::ssize_t i=0;i<coords.shape(0);++i) {
      // Difference and integer raster products are bounded before arithmetic.
      for (int j=0;j<4;++j) if (coords(i,j)<-1000000000LL || coords(i,j)>1000000000LL)
        throw std::invalid_argument("grid coordinate exceeds supported range");
      if (!std::isfinite(heights(i,0)) || !std::isfinite(heights(i,1)))
        throw std::invalid_argument("nonfinite ray height");
      const I dr=coords(i,2)-coords(i,0), dc=coords(i,3)-coords(i,1);
      const I steps=std::max(std::abs(dr),std::abs(dc));
      for (I k=1;k<steps;++k) {
        const I row=coords(i,0)+floor_div(k*dr+steps/2,steps);
        const I col=coords(i,1)+floor_div(k*dc+steps/2,steps);
        const double altitude=heights(i,0)+(double(k)/double(steps))*(heights(i,1)-heights(i,0));
        if (altitude<min_height || altitude>max_height || row<0 || row>=height || col<0 || col>=width) continue;
        const I layer=std::min(floor_index((altitude-min_height)/height_resolution),depth-1);
        const I key=(row*width+col)*depth+layer;
        const std::uint64_t bit=std::uint64_t(1)<<(key%64);
        auto& word=seen[key/64];
        if (!(word&bit)) { word|=bit; keys.push_back(key); }
      }
    }
    std::sort(keys.begin(),keys.end());
  }
  py::array_t<I> result(keys.size());
  std::copy(keys.begin(),keys.end(),result.mutable_data());
  return result;
}

PYBIND11_MODULE(_ray_native,m) {
  m.doc()="Ray evidence kernels with GIL-free traversal and cached voxel membership";
  m.attr("API_VERSION")=1;
  py::class_<VoxelIndex>(m,"VoxelIndex")
      .def(py::init<>())
      .def("add",[](VoxelIndex& index,Ints keys){index.change(keys,true);})
      .def("erase",[](VoxelIndex& index,Ints keys){index.change(keys,false);})
      .def("clear",&VoxelIndex::clear);
  m.def("voxel_evidence",&voxel_evidence);
  m.def("grid_free_keys",&grid_free_keys);
  m.def("select_grouped",&select_grouped);
}
