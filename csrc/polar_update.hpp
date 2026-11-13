// Polar planar free-space evidence (RayOccupancy2D free_space_mode="polar").
// Mirrors RayOccupancy2D._polar_update / _polar_rays exactly; tests compare both
// backends cell by cell. Runs after grid_update on the same observation.

void polar_update(Floats rays, Floats distances, Scores scores, GridValues values,
                  Scores counts, double origin_x, double origin_y, double resolution,
                  double min_height, double max_height, double max_range, double planar_range,
                  I min_observations, I polar_bins, double closest_margin) {
  if (rays.ndim() != 2 || rays.shape(1) != 6 || distances.ndim() != 1 ||
      distances.shape(0) != rays.shape(0) || scores.ndim() != 3 || values.ndim() != 2 ||
      counts.ndim() != 2 || scores.shape(0) != values.shape(0) ||
      scores.shape(1) != values.shape(1) || counts.shape(0) != values.shape(0) ||
      counts.shape(1) != values.shape(1) || scores.shape(2) <= 0 || !scores.writeable() ||
      !values.writeable() || !counts.writeable())
    throw std::invalid_argument("invalid polar update buffers");
  for (double value : {origin_x, origin_y, resolution, min_height, max_height, max_range,
                       planar_range, closest_margin})
    if (!std::isfinite(value))
      throw std::invalid_argument("nonfinite polar configuration");
  if (resolution <= 0 || max_range <= 0 || planar_range <= 0 || closest_margin < 0 ||
      min_observations < 1 || min_observations > 127 || polar_bins < 1)
    throw std::invalid_argument("invalid polar configuration");
  const I height = scores.shape(0), width = scores.shape(1), depth = scores.shape(2);
  if (width > std::numeric_limits<I>::max() / height / depth)
    throw std::invalid_argument("grid dimensions exceed supported range");
  auto r = rays.unchecked<2>();
  auto d = distances.unchecked<1>();
  const auto* score = scores.data();
  auto* value = values.mutable_data();
  auto* count = counts.mutable_data();
  const I columns = height * width;
  py::gil_scoped_release release;

  auto row_of = [&](double y) { return grid_coordinate((y - origin_y) / resolution); };
  auto col_of = [&](double x) { return grid_coordinate((x - origin_x) / resolution); };
  auto positive = [&](I column) {
    for (I layer = 0; layer < depth; ++layer)
      if (score[column * depth + layer] > 0)
        return true;
    return false;
  };

  // 1. Current in-band hits reset planar counts and veto planar clearing.
  std::vector<std::uint8_t> hit(columns, 0);
  for (py::ssize_t i = 0; i < r.shape(0); ++i) {
    for (int j = 0; j < 6; ++j)
      if (!std::isfinite(r(i, j)))
        throw std::invalid_argument("nonfinite polar ray");
    if (!(d(i) <= max_range) || r(i, 5) < min_height || r(i, 5) > max_height)
      continue;
    const I row = row_of(r(i, 4)), col = col_of(r(i, 3));
    if (row >= 0 && row < height && col >= 0 && col < width) {
      hit[row * width + col] = 1;
      count[row * width + col] = 0;
    }
  }

  // 2. In-band rays, nearest planar return per (origin group, sector).
  struct Beam {
    double ox, oy, ex, ey, planar;
    std::int64_t key;
  };
  std::vector<Beam> beams;
  beams.reserve(r.shape(0));
  constexpr double kPi = 3.14159265358979323846;
  for (py::ssize_t i = 0; i < r.shape(0); ++i) {
    if (r(i, 2) < min_height || r(i, 2) > max_height || r(i, 5) < min_height ||
        r(i, 5) > max_height || !(d(i) <= max_range))
      continue;
    const double dx = r(i, 3) - r(i, 0), dy = r(i, 4) - r(i, 1);
    const double planar = std::hypot(dx, dy);
    if (!(planar > 1e-6))
      continue;
    const double bearing = std::atan2(dy, dx);
    I bin = static_cast<I>(std::floor((bearing + kPi) / (2 * kPi) * double(polar_bins)));
    bin = ((bin % polar_bins) + polar_bins) % polar_bins;
    std::int64_t key = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const double rounded = std::nearbyint(r(i, axis) / 0.25);
      if (!(rounded >= -16384.0 && rounded < 16384.0))
        throw std::invalid_argument("ray origin outside the supported +-4 km range");
      key = (key << 15) | (static_cast<std::int64_t>(rounded) + (1 << 14));
    }
    beams.push_back({r(i, 0), r(i, 1), r(i, 3), r(i, 4), planar, key * polar_bins + bin});
  }
  std::unordered_map<std::int64_t, double> nearest;
  nearest.reserve(beams.size());
  for (const auto& beam : beams) {
    auto inserted = nearest.emplace(beam.key, beam.planar);
    if (!inserted.second && beam.planar < inserted.first->second)
      inserted.first->second = beam.planar;
  }

  // 3. Rasterize up to the range limit; stop at the first known obstacle.
  std::vector<std::uint8_t> blocked(columns, 0), seen(columns, 0);
  for (auto& beam : beams) {
    const double limit = nearest[beam.key];
    double ex = beam.ex, ey = beam.ey;
    if (beam.planar > limit + closest_margin) {
      const double ratio = limit / beam.planar;
      ex = beam.ox + (beam.ex - beam.ox) * ratio;
      ey = beam.oy + (beam.ey - beam.oy) * ratio;
    }
    const double planar = std::sqrt((ex - beam.ox) * (ex - beam.ox) + (ey - beam.oy) * (ey - beam.oy));
    const double scale = std::min(1.0, planar_range / planar);
    ex = beam.ox + (ex - beam.ox) * scale;
    ey = beam.oy + (ey - beam.oy) * scale;
    const I row0 = row_of(beam.oy), col0 = col_of(beam.ox), row1 = row_of(ey), col1 = col_of(ex);
    const I dr = row1 - row0, dc = col1 - col0;
    const I steps = std::max(std::abs(dr), std::abs(dc));
    for (I k = 1; k < steps; ++k) {
      const I row = row0 + floor_div(k * dr + steps / 2, steps);
      const I col = col0 + floor_div(k * dc + steps / 2, steps);
      if (row < 0 || row >= height || col < 0 || col >= width)
        continue;
      const I column = row * width + col;
      if (!blocked[column])
        blocked[column] = (hit[column] || positive(column)) ? 2 : 1;
      if (blocked[column] == 2)
        break;
      seen[column] = 1;
    }
  }
  for (I column = 0; column < columns; ++column)
    if (seen[column])
      count[column] = static_cast<std::int8_t>(std::min<I>(count[column] + 1, min_observations));

  // 4. The height update may have reset earlier planar-free columns anywhere.
  for (I column = 0; column < columns; ++column)
    if (count[column] >= min_observations && value[column] != 0 && value[column] != 100 &&
        !positive(column))
      value[column] = 0;
}
