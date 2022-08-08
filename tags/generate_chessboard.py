# 生成 apriltags 组成的棋盘格, 用于相机标定
import numpy as np
import cv2
from pathlib import Path


if __name__ == '__main__':
    # tag 存放目录
    CurDir = Path(__file__).resolve().parent
    TagFile = str(CurDir) + '\\tag36h11\\tag36_11_{:05d}.png'

    # chessboard 设置
    # 对于 tag36, 内部编码区为 (6, 6), 而检测的角点为黑边角点为 (8, 8), 总尺寸还包括外部白边 (10, 10)
    tag_size = 10
    grid_size = 12  # 棋盘格中放置一个 tag 的格子尺寸, 需要 > 10
    bias = (grid_size - tag_size) // 2  # 棋盘格左上角到格子左上角的偏移量
    row, col = 3, 4  # 棋盘格尺寸
    chessboard = np.full((row * grid_size, col * grid_size, 3), 255, dtype=np.uint8)
    scale = 20  # 图片整体放大倍数

    # 按顺序将 tag 读入, 填入每个棋盘格
    for i in range(row):
        for j in range(col):
            drow, dcol = i * grid_size + bias, j * grid_size + bias
            tag_img = cv2.imread(TagFile.format(i*col + j))
            chessboard[drow : drow+tag_size, dcol : dcol+tag_size] = tag_img

    # 整体放大
    chessboard = cv2.resize(chessboard, np.array(chessboard.shape[1::-1]) * scale, interpolation=cv2.INTER_NEAREST )

    # 记录每个角点的像素坐标
    # 以第一个 tag 的左上角为原点 (col, row)，打印后根据真实尺寸缩放到真实坐标
    # 针对每个 tag，角点顺序为 (左下, 右下, 右上, 左上), 与 apriltag 检测顺序一致
    corners = []
    border_size = (tag_size-2) * scale  # 放大后的黑框尺寸 (8, 8)*20
    grid_size = grid_size * scale  # 放大后的 grid_size
    tmp = np.array([[0, border_size], [border_size, border_size], [border_size, 0], [0, 0]], dtype=np.int16)
    for i in range(row):
        for j in range(col):
            dcol, drow = j * grid_size, i * grid_size  # 偏移量
            corners.append(tmp + (dcol, drow))
            cv2.putText(chessboard, str(i*col + j), (dcol+5, drow+border_size), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (125, 125, 125), 2)
    corners = np.concatenate(corners, axis=0)

    # 绘制每一个角点并可视化
    chessboard_show = chessboard.copy()
    cv2.drawChessboardCorners(chessboard_show, (4, corners.shape[0]//4),
        (corners + (bias+1)*scale).astype(np.float32) , True)
    cv2.namedWindow("chessboard")
    cv2.imshow("chessboard", chessboard_show)
    cv2.waitKey(0)

    # 将角点像素坐标按照黑框长度归一化, 1 代表黑框长度
    corners = corners.astype(np.float32) / border_size

    # 保存
    cv2.imwrite(str(CurDir / "apriltag_board.png"), chessboard)
    np.savez(str(CurDir / "apriltag_corners.npz"), corners=corners)

    # 读取
    data = np.load(str(CurDir / "apriltag_corners.npz"))
    print(data["corners"])