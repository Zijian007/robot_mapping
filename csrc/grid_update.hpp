// In-place height-binned evidence update. Python owns grid allocation and holds
// the OnlineMap lock; this kernel does not retain pointers to NumPy buffers.
#include <optional>
#include <pybind11/stl.h>

using Scores = py::array_t<std::int8_t, py::array::c_style>;
using GridValues = py::array_t<std::int16_t, py::array::c_style>;
using Footprint = std::optional<std::array<double, 2>>;

struct GridRay {
  I row0, col0, row1, col1;
  double z0, z1, length;
  // Samples at or beyond this fraction of the beam add no miss evidence (end margin).
  double free_until;
  bool planar;
};

I grid_coordinate(double value) {
  const I result = floor_index(value);
  if (result < -1000000000LL || result > 1000000000LL)
    throw std::invalid_argument("grid coordinate exceeds supported range");
  return result;
}

void grid_update(Floats rays, Floats distances, Ints groups, Scores scores, GridValues values,
                 double origin_x, double origin_y, double resolution, double min_height,
                 double max_height, double height_resolution, double max_range, I max_rays,
                 double free_radius, Footprint previous, Footprint current,
                 std::optional<Scores> projected_counts, I projected_min_observations,
                 std::optional<double> projected_max_range, int score_hit, int score_miss,
                 int score_min, int score_max, int score_occupied, double free_end_margin) {
  const double planar_range = projected_max_range.value_or(max_range);
  if (!std::isfinite(planar_range) || planar_range <= 0 || projected_min_observations < 1 ||
      projected_min_observations > 127)
    throw std::invalid_argument("invalid projected free-space parameters");
  // Same bounds as ray_grid.validate_score_rules; scores are stored as int8.
  if (score_hit < 1 || score_hit > 127 || score_miss < 1 || score_miss > 127 || score_min < -128 ||
      score_min > -1 || score_occupied < 1 || score_occupied > score_max || score_max > 127)
    throw std::invalid_argument("invalid grid score rules");
  if (rays.ndim() != 2 || rays.shape(1) != 6 || distances.ndim() != 1 ||
      distances.shape(0) != rays.shape(0) || groups.ndim() != 2 || groups.shape(1) != 5 ||
      groups.shape(0) != rays.shape(0) || scores.ndim() != 3 || values.ndim() != 2 ||
      scores.shape(0) != values.shape(0) || scores.shape(1) != values.shape(1) ||
      scores.shape(0) <= 0 || scores.shape(1) <= 0 || scores.shape(2) <= 0 || !scores.writeable() ||
      !values.writeable())
    throw std::invalid_argument("invalid grid update buffers");
  for (double value : {origin_x, origin_y, resolution, min_height, max_height, height_resolution,
                       max_range, free_radius, free_end_margin})
    if (!std::isfinite(value))
      throw std::invalid_argument("nonfinite grid configuration");
  if (free_end_margin < 0)
    throw std::invalid_argument("free_end_margin must be nonnegative");
  if (resolution <= 0 || height_resolution <= 0 || max_height < min_height || max_range <= 0 ||
      max_rays < 1)
    throw std::invalid_argument("invalid grid update configuration");
  const I height = scores.shape(0), width = scores.shape(1), depth = scores.shape(2);
  if (projected_counts && (projected_counts->ndim() != 2 || projected_counts->shape(0) != height ||
                           projected_counts->shape(1) != width || !projected_counts->writeable()))
    throw std::invalid_argument("invalid projected free-space buffer");
  if (width > std::numeric_limits<I>::max() / height / depth)
    throw std::invalid_argument("grid dimensions exceed supported range");
  auto r = rays.unchecked<2>();
  auto d = distances.unchecked<1>();
  auto g = groups.unchecked<2>();
  auto* score = scores.mutable_data();
  auto* value = values.mutable_data();
  auto* planar_count = projected_counts ? projected_counts->mutable_data() : nullptr;
  py::gil_scoped_release release;

  // Complete validation and geometric preparation before touching evidence.
  const I radius = std::max<I>(1, grid_coordinate(std::ceil(free_radius / resolution)));
  auto footprint_cell = [&](const Footprint& xy) -> std::array<I, 2> {
    return xy ? std::array<I, 2>{grid_coordinate(((*xy)[1] - origin_y) / resolution),
                                 grid_coordinate(((*xy)[0] - origin_x) / resolution)}
              : std::array<I, 2>{0, 0};
  };
  const auto prev = footprint_cell(previous), now = footprint_cell(current);
  std::unordered_map<Group, I, GroupHash> nearest;
  nearest.reserve(r.shape(0));
  std::vector<I> hits;
  hits.reserve(r.shape(0));
  for (py::ssize_t i = 0; i < r.shape(0); ++i) {
    for (int j = 0; j < 6; ++j)
      if (!std::isfinite(r(i, j)))
        throw std::invalid_argument("nonfinite grid ray");
    if (!std::isfinite(d(i)) || d(i) <= 1e-6)
      throw std::invalid_argument("grid ray lengths must exceed 1e-6");
    const Group key{g(i, 0), g(i, 1), g(i, 2), g(i, 3), g(i, 4)};
    auto inserted = nearest.emplace(key, i);
    if (!inserted.second && d(i) < d(inserted.first->second))
      inserted.first->second = i;
    if (d(i) <= max_range && r(i, 5) >= min_height && r(i, 5) <= max_height) {
      const I row = grid_coordinate((r(i, 4) - origin_y) / resolution);
      const I col = grid_coordinate((r(i, 3) - origin_x) / resolution);
      const I layer = std::min(floor_index((r(i, 5) - min_height) / height_resolution), depth - 1);
      if (row >= 0 && row < height && col >= 0 && col < width)
        hits.push_back((row * width + col) * depth + layer);
    }
  }
  std::vector<std::pair<Group, I>> ordered(nearest.begin(), nearest.end());
  std::sort(ordered.begin(), ordered.end());
  const I selected = std::min<I>(max_rays, ordered.size());
  std::vector<GridRay> beams;
  beams.reserve(selected);
  // Match np.linspace(0, n-1, limit, dtype=int), including its final endpoint.
  const double spacing = selected > 1 ? double(ordered.size() - 1) / double(selected - 1) : 0.;
  for (I k = 0; k < selected; ++k) {
    const I index =
        (k == selected - 1 && selected > 1) ? ordered.size() - 1 : static_cast<I>(k * spacing);
    const I i = ordered[index].second;
    const double scale = std::min(1., max_range / d(i));
    const double x = r(i, 0) + (r(i, 3) - r(i, 0)) * scale;
    const double y = r(i, 1) + (r(i, 4) - r(i, 1)) * scale;
    const double z = r(i, 2) + (r(i, 5) - r(i, 2)) * scale;
    beams.push_back({grid_coordinate((r(i, 1) - origin_y) / resolution),
                     grid_coordinate((r(i, 0) - origin_x) / resolution),
                     grid_coordinate((y - origin_y) / resolution),
                     grid_coordinate((x - origin_x) / resolution), r(i, 2), z,
                     std::min(d(i), max_range), free_until(d(i), max_range, free_end_margin),
                     r(i, 2) >= min_height && r(i, 2) <= max_height && r(i, 5) >= min_height &&
                         r(i, 5) <= max_height});
  }
  const I columns = height * width, count = columns * depth;
  std::vector<std::uint64_t> seen(count / 64 + 1, 0), dirty_bits(columns / 64 + 1, 0);
  std::vector<I> dirty;
  std::vector<std::uint64_t> planar_hits(planar_count ? columns / 64 + 1 : 0, 0);
  auto first = [](std::vector<std::uint64_t>& bits, I key) {
    const auto mask = std::uint64_t(1) << (key % 64);
    auto& word = bits[key / 64];
    if (word & mask)
      return false;
    word |= mask;
    return true;
  };
  auto touch = [&](I column) {
    if (first(dirty_bits, column))
      dirty.push_back(column);
  };
  // Mark all hits first: they win over free traversal in this observation.
  for (I key : hits)
    if (first(seen, key)) {
      score[key] = static_cast<std::int8_t>(std::min<int>(score[key] + score_hit, score_max));
      touch(key / depth);
      if (planar_count) {
        first(planar_hits, key / depth);
        planar_count[key / depth] = 0;
      }
    }
  for (const auto& beam : beams) {
    const I dr = beam.row1 - beam.row0, dc = beam.col1 - beam.col0;
    const I steps = std::max(std::abs(dr), std::abs(dc));
    for (I k = 1; k < steps; ++k) {
      if (double(k) / double(steps) >= beam.free_until)
        break;
      const I row = beam.row0 + floor_div(k * dr + steps / 2, steps);
      const I col = beam.col0 + floor_div(k * dc + steps / 2, steps);
      const double altitude = beam.z0 + (double(k) / double(steps)) * (beam.z1 - beam.z0);
      if (altitude < min_height || altitude > max_height || row < 0 || row >= height || col < 0 ||
          col >= width)
        continue;
      const I layer = std::min(floor_index((altitude - min_height) / height_resolution), depth - 1);
      const I key = (row * width + col) * depth + layer;
      if (first(seen, key)) {
        score[key] = static_cast<std::int8_t>(std::max<int>(score[key] - score_miss, score_min));
        touch(key / depth);
      }
    }
  }
  if (planar_count) {
    std::vector<std::uint64_t> planar_seen(columns / 64 + 1, 0);
    // Cache all-height occlusion queries for the current observation only.
    std::vector<std::uint8_t> blocked(columns, 0);
    for (const auto& beam : beams) {
      if (!beam.planar)
        continue;
      const I dr = beam.row1 - beam.row0, dc = beam.col1 - beam.col0;
      const I steps = std::max(std::abs(dr), std::abs(dc));
      for (I k = 1; k < steps; ++k) {
        if ((double(k) / double(steps)) * beam.length >= planar_range)
          break;
        const I row = beam.row0 + floor_div(k * dr + steps / 2, steps);
        const I col = beam.col0 + floor_div(k * dc + steps / 2, steps);
        if (row < 0 || row >= height || col < 0 || col >= width)
          continue;
        const I column = row * width + col;
        if (!blocked[column]) {
          bool positive = (planar_hits[column / 64] >> (column % 64)) & 1;
          for (I layer = 0; layer < depth && !positive; ++layer)
            positive = score[column * depth + layer] > 0;
          blocked[column] = positive ? 2 : 1;
        }
        if (blocked[column] == 2)
          break;
        if (first(planar_seen, column)) {
          planar_count[column] = std::min<I>(planar_count[column] + 1, projected_min_observations);
          touch(column);
        }
      }
    }
  }
  // Refresh the former prior's whole bounding patch, exactly as the reference.
  if (previous)
    for (I row = std::max<I>(0, prev[0] - radius); row < std::min(height, prev[0] + radius + 1);
         ++row)
      for (I col = std::max<I>(0, prev[1] - radius); col < std::min(width, prev[1] + radius + 1);
           ++col)
        touch(row * width + col);
  for (I column : dirty) {
    bool occupied = false, free = true, positive = false;
    for (I layer = 0; layer < depth; ++layer) {
      const int evidence = score[column * depth + layer];
      occupied = occupied || evidence >= score_occupied;
      free = free && evidence < 0;
      positive = positive || evidence > 0;
    }
    if (planar_count) {
      if (positive)
        planar_count[column] = 0;
      free = !positive && planar_count[column] >= projected_min_observations;
    }
    value[column] = occupied ? 100 : (free ? 0 : -1);
  }
  if (current)
    for (I row = std::max<I>(0, now[0] - radius); row < std::min(height, now[0] + radius + 1);
         ++row)
      for (I col = std::max<I>(0, now[1] - radius); col < std::min(width, now[1] + radius + 1);
           ++col) {
        if ((row - now[0]) * (row - now[0]) + (col - now[1]) * (col - now[1]) > radius * radius)
          continue;
        const I column = row * width + col;
        if (value[column] != -1)
          continue;
        bool positive = planar_count && ((planar_hits[column / 64] >> (column % 64)) & 1);
        for (I layer = 0; layer < depth; ++layer)
          positive = positive || score[column * depth + layer] > 0;
        if (!positive)
          value[column] = 0;
      }
}
