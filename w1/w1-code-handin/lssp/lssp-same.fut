-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1i32, -2i32, -2i32, 0i32, 0i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    5i32
-- }
-- compiled input {
--    empty([0]i32)
-- }
-- output {
--    0i32
-- }
-- compiled input {
--    [1i32, 2i32, 3i32, 4i32, 5i32]
-- }
-- output {
--    1i32
-- }
-- compiled input {
--    [0i32, 0i32, 0i32, 1, 0i32, 0i32, 1, 0i32]
-- }
-- output {
--    3i32
-- }


import "lssp"
import "lssp-seq"

let main (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
