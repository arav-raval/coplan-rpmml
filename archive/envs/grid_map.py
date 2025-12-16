def ascii_to_segments(grid, cell=20):
    """
    grid: list of strings where '#' is wall, '.' is free.
    Returns list of wall segments ((x1,y1),(x2,y2)).
    """
    H, W = len(grid), len(grid[0])
    segs = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != '#': continue
            x, y = c*cell, (H-1-r)*cell
            segs += [((x, y), (x+cell, y)),
                     ((x+cell, y), (x+cell, y+cell)),
                     ((x+cell, y+cell), (x, y+cell)),
                     ((x, y+cell), (x, y))]
    return segs
