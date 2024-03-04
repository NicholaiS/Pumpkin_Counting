import cv2

chunkpos = [(0,0), (10,0), (20,0)]
pumpkin_positions_chunk = [10,0]

for chunk_pos in chunkpos:
    print(chunk_pos)
    
    # Adjust pumpkin positions with chunk coordinates
    adjusted_pumpkin_positions = [(x + chunk_pos[0], y + chunk_pos[1]) for (x, y) in pumpkin_positions_chunk]
                
    # Log the adjusted positions in pumpkin_positions
    for pos in adjusted_pumpkin_positions:
        pumpkin_positions.setdefault(pos, True)


    for pos in pumpkin_positions.keys():
        cv2.circle(field, pos, 5, (0, 0, 255), 1)