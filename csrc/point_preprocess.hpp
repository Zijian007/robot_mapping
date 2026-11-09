// Point compaction and field copying. Coordinate matrix arithmetic and all
// timestamped robot-transform policy remain with the Python caller.
template <typename T> using PointArray = py::array_t<T, 0>;

// Small read-only predicates avoid NumPy's per-call broadcasting/allclose
// setup. Gram products and determinants are still computed by NumPy so their
// rounding and the order of rejection checks remain identical to the oracle.
bool rigid_structure_valid(const Floats& matrix) {
  if (matrix.ndim()!=2 || matrix.shape(0)!=4 || matrix.shape(1)!=4) return false;
  auto m=matrix.unchecked<2>();
  for (int i=0;i<4;++i) for (int j=0;j<4;++j)
    if (!std::isfinite(m(i,j))) return false;
  for (int j=0;j<4;++j) {
    const double target=j==3 ? 1. : 0.;
    if (std::abs(m(3,j)-target)>1e-8+1e-5*target) return false;
  }
  return true;
}

bool rigid_gram_valid(const Floats& gram) {
  if (gram.ndim()!=2 || gram.shape(0)!=3 || gram.shape(1)!=3) return false;
  auto g=gram.unchecked<2>();
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    const double target=i==j ? 1. : 0.;
    if (!std::isfinite(g(i,j)) || std::abs(g(i,j)-target)>1e-5+1e-5*target) return false;
  }
  return true;
}

bool rigid_determinant_valid(double determinant) {
  return std::isfinite(determinant) && std::abs(determinant-1.)<=1e-5+1e-5;
}

py::array_t<float> prepare_points(PointArray<float> xyz, PointArray<float> intensity,
                                 PointArray<I> offsets, double blind_squared, bool include_origins) {
  if (xyz.ndim()!=2 || xyz.shape(1)!=3 || intensity.ndim()!=1 || offsets.ndim()!=1 ||
      intensity.shape(0)!=xyz.shape(0) || offsets.shape(0)!=xyz.shape(0) ||
      !std::isfinite(blind_squared) || blind_squared<0)
    throw std::invalid_argument("invalid scan preparation arguments");
  auto p=xyz.unchecked<2>(); auto field=intensity.unchecked<1>(); auto stamp=offsets.unchecked<1>();
  std::vector<I> kept;
  {
    py::gil_scoped_release release;
    kept.reserve(p.shape(0));
    for (I i=0;i<p.shape(0);++i) {
      const double x=p(i,0), y=p(i,1), z=p(i,2);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
          !std::isfinite(field(i)) || stamp(i)<0)
        throw std::invalid_argument("invalid scan point fields");
      if ((x*x+y*y)+z*z>blind_squared) kept.push_back(i);
    }
  }
  py::array_t<float> result({static_cast<py::ssize_t>(kept.size()), py::ssize_t(include_origins ? 8 : 5)});
  auto out=result.mutable_unchecked<2>();
  {
    py::gil_scoped_release release;
    for (I row=0;row<static_cast<I>(kept.size());++row) {
      const I i=kept[row];
      for (int j=0;j<3;++j) out(row,j)=p(i,j);
      out(row,3)=field(i);
      out(row,4)=static_cast<double>(stamp(i))*1e-6;
      if (include_origins) for (int j=5;j<8;++j) out(row,j)=0.f;
    }
  }
  return result;
}

template <typename T>
py::tuple compact_aabbs(PointArray<T> points, const Floats& boxes, bool weak_scalars) {
  auto p=points.template unchecked<2>(); auto bounds=boxes.unchecked<3>();
  std::vector<I> kept;
  {
    py::gil_scoped_release release;
    std::vector<std::array<double,6>> comparisons;
    comparisons.reserve(bounds.shape(0));
    for (I b=0;b<bounds.shape(0);++b) for (int j=0;j<3;++j)
      if (!std::isfinite(bounds(b,0,j)) || !std::isfinite(bounds(b,1,j)) || bounds(b,0,j)>bounds(b,1,j))
        throw std::invalid_argument("invalid collision bounds");
    for (I b=0;b<bounds.shape(0);++b) {
      std::array<double,6> box;
      for (int side=0;side<2;++side) for (int j=0;j<3;++j) {
        double value=bounds(b,side,j);
        // NumPy 1.x weak scalar promotion: float32 arrays compare ordinary
        // float64 scalar bounds after rounding those scalars to float32.
        if (weak_scalars && sizeof(T)==sizeof(float) && std::abs(value)<3.4e38)
          value=static_cast<float>(value);
        box[side*3+j]=value;
      }
      comparisons.push_back(box);
    }
    kept.reserve(p.shape(0));
    for (I i=0;i<p.shape(0);++i) {
      const double x=p(i,0), y=p(i,1), z=p(i,2);
      bool inside=false;
      for (const auto& box : comparisons)
        if (x>=box[0] && x<=box[3] && y>=box[1] && y<=box[4] && z>=box[2] && z<=box[5])
          { inside=true; break; }
      if (!inside) kept.push_back(i);
    }
  }
  const I removed=p.shape(0)-kept.size();
  if (!removed) return py::make_tuple(points,0);
  py::array_t<T> result({static_cast<py::ssize_t>(kept.size()),p.shape(1)});
  auto out=result.template mutable_unchecked<2>();
  {
    py::gil_scoped_release release;
    for (I row=0;row<static_cast<I>(kept.size());++row)
      for (I j=0;j<p.shape(1);++j) out(row,j)=p(kept[row],j);
  }
  return py::make_tuple(result,removed);
}

py::tuple filter_aabbs(py::array points,Floats boxes,bool weak_scalars) {
  if (points.ndim()!=2 || points.shape(1)<3 || boxes.ndim()!=3 || boxes.shape(1)!=2 || boxes.shape(2)!=3)
    throw std::invalid_argument("collision filtering needs (N,>=3) points and (B,2,3) bounds");
  if (points.dtype().is(py::dtype::of<float>()))
    return compact_aabbs(py::reinterpret_borrow<PointArray<float>>(points),boxes,weak_scalars);
  if (points.dtype().is(py::dtype::of<double>()))
    return compact_aabbs(py::reinterpret_borrow<PointArray<double>>(points),boxes,weak_scalars);
  throw std::invalid_argument("collision filtering requires float32 or float64 points");
}

py::tuple merge_prepared(PointArray<float> front,PointArray<float> back,
                         Floats xyz,Floats origins,float dt_ms,float end_ms) {
  if (front.ndim()!=2 || back.ndim()!=2 || front.shape(1)!=back.shape(1) || back.shape(1)<5 ||
      xyz.ndim()!=2 || xyz.shape(0)!=back.shape(0) || xyz.shape(1)!=3 ||
      origins.ndim()!=2 || origins.shape(1)!=3 ||
      origins.shape(0)!=(back.shape(1)>=8 ? back.shape(0) : 0))
    throw std::invalid_argument("invalid prepared scan merge buffers");
  auto a=front.unchecked<2>(); auto b=back.unchecked<2>();
  auto p=xyz.unchecked<2>(); auto origin=origins.unchecked<2>();
  std::vector<I> kept;
  {
    py::gil_scoped_release release;
    kept.reserve(b.shape(0));
    for (I i=0;i<b.shape(0);++i) {
      const float time=b(i,4)+dt_ms;
      if (time>=0.f && time<=end_ms) kept.push_back(i);
    }
  }
  const I removed=b.shape(0)-kept.size();
  if (kept.empty()) return py::make_tuple(front,removed,0);
  py::array_t<float> result({a.shape(0)+static_cast<py::ssize_t>(kept.size()),a.shape(1)});
  auto out=result.mutable_unchecked<2>();
  {
    py::gil_scoped_release release;
    for (I i=0;i<a.shape(0);++i) for (I j=0;j<a.shape(1);++j) out(i,j)=a(i,j);
    for (I row=0;row<static_cast<I>(kept.size());++row) {
      const I i=kept[row], target=a.shape(0)+row;
      for (I j=0;j<b.shape(1);++j) out(target,j)=b(i,j);
      for (int j=0;j<3;++j) out(target,j)=p(i,j);
      if (b.shape(1)>=8) for (int j=0;j<3;++j) out(target,5+j)=origin(i,j);
      out(target,4)=b(i,4)+dt_ms;
    }
  }
  return py::make_tuple(result,removed,kept.size());
}
