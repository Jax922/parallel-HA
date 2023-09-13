-----------------------------------------
-- Parallel Longest Satisfying Segment --
-- Longest Satisfying Segment          --
-- ASSIGNMENT 1: fill in the blanks    --
--       See lecture notes             --
-- pred1 x   = p [x]                   --
-- pred2 x y = p [x,y]                 --
-----------------------------------------

type int = i32
let max (x:i32, y:i32) = i32.max x y

-- the task is to implement this operator by filling in the blanks
let redOp (pred2 : i32 -> i32 -> bool)
          (lssx: i32, lisx: i32, lcsx: i32, tlx: i32, firstx: i32, lastx: i32)
          (lssy: i32, lisy: i32, lcsy: i32, tly: i32, firsty: i32, lasty: i32)
        : (i32,i32,i32,i32,i32,i32) =

  let connect= pred2 lastx firsty 
  let newlss = max (max (lssx, lssy), if connect then lcsx + lisy else max (lcsx, lisy))
  let newlis = if (connect && tlx == lisx) then lisx + lisy else lisx
  let newlcs = if (connect && tly == lcsy) then lcsx + lcsy else lcsy
  let newtl  = tlx + tly 
  let first  = if tlx == 0 then firsty else firstx
  let last   = if tly == 0 then lastx else lasty in
  (newlss, newlis, newlcs, newtl, first, last)

let mapOp (pred1 : i32 -> bool) (x: i32) : (i32,i32,i32,i32,i32,i32) =
  let xmatch = if pred1 x then 1 else 0 in
  (xmatch, xmatch, xmatch, 1, x, x)

let lssp (pred1 : i32 -> bool) 
         (pred2 : i32 -> i32 -> bool) 
         (xs    : []i32) : i32 =
  let (x,_,_,_,_,_) =
        map (mapOp pred1) xs
     |> reduce (redOp pred2) (0,0,0,0,0,0)
  in  x
