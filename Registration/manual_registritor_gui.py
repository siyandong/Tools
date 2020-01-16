import numpy as np
import cv2
from tkinter import *
from PIL import Image, ImageTk
import rgbd_calib
from kabsch import Kabsch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator


# params
g_enable_calib = True # kinect color and depth alignment
g_color_dir1 = 'D:/data/7-scenes/total/chess/seq-03/frame-000050.color.png'
g_color_dir2 = 'D:/data/7-scenes/total/chess/seq-03/frame-000150.color.png'
g_depth_dir1 = 'D:/data/7-scenes/total/chess/seq-03/frame-000050.depth.png'
g_depth_dir2 = 'D:/data/7-scenes/total/chess/seq-03/frame-000150.depth.png'
g_camera_factor = 1000.
g_cx = 320.
g_cy = 240.
g_fx = 585.
g_fy = 585.
g_img_rows = 480
g_img_cols = 640

# params fixed
g_color_img1 = cv2.imread(g_color_dir1)
g_color_img2 = cv2.imread(g_color_dir2)
g_depth_img1 = cv2.imread(g_depth_dir1, -1)
g_depth_img2 = cv2.imread(g_depth_dir2, -1)
if g_enable_calib:
    g_color_img1 = rgbd_calib.color_calibration(g_color_img1)
    g_color_img2 = rgbd_calib.color_calibration(g_color_img2)
g_vis_img1 = g_color_img1.copy()
g_vis_img2 = g_color_img2.copy()
g_rand_color = (0, 0, 255)
g_crt_match1 = (0, 0)
g_crt_match2 = (0, 0)
g_corres_l1 = []
g_corres_l2 = []
g_p3d_l1 = []
g_p3d_l2 = []
g_intrinsic = np.matrix([[g_fx, 0, g_cx], [0, g_fy, g_cy], [0, 0, 1]])


# generate pointcloud from depth and pose
def generate_pc(img_depth, sample_step=16, Rt=np.eye(4, dtype=float)):
    l_p3d = []
    for r in range(0, img_depth.shape[0], sample_step):
        for c in range(0, img_depth.shape[1], sample_step):
            depth = img_depth[r,c]
            if depth==0 or depth==65535:
                continue
            if depth>5000:
                print('depth', depth)
            z = depth / g_camera_factor
            x = float(c-g_cx) * z / g_fx
            y = float(r-g_cy) * z / g_fy
            p3d = np.array([x, y, z, 1.0])
            p3d = np.dot(Rt, p3d)
            l_p3d.append(p3d)
    return l_p3d

# generate pointcloud from depth and pose
def generate_pc_color(img_depth, img_color, sample_step=16, Rt=np.eye(4, dtype=float)):
    l_p3d = []
    for r in range(0, img_depth.shape[0], sample_step):
        for c in range(0, img_depth.shape[1], sample_step):
            depth = img_depth[r,c]
            if depth==0 or depth==65535:
                continue
            if depth>5000:
                print('depth', depth)
            z = depth / g_camera_factor
            x = float(c-g_cx) * z / g_fx
            y = float(r-g_cy) * z / g_fy
            p3d = np.array([x, y, z, 1.0])
            p3d = np.dot(Rt, p3d)
            color = img_color[r, c]/255.
            l_p3d.append(np.array([p3d[0], p3d[1], p3d[2], color[2], color[1], color[0]]))
    return l_p3d

# result visualization
def draw_merged_pc(pc1, pc2):
    x1 = pc1[:, 0]  
    y1 = pc1[:, 1]  
    z1 = pc1[:, 2]  
    x2 = pc2[:, 0]
    y2 = pc2[:, 1]
    z2 = pc2[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x1, y1, z1, s=6., c='r', label='pc1')
    ax.scatter(x2, y2, z2, s=6., c='b', label='pc1')
    ax.legend(loc='best')
    plt.show()
    return

# result visualization
def draw_merged_pc_color(pc1, pc2):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pc1[:,0], pc1[:,1], pc1[:,2], s=6., c=pc1[:,3:6], marker='+', label='pc1')
    ax.scatter(pc2[:,0], pc2[:,1], pc2[:,2], s=6., c=pc2[:,3:6], marker='o', label='pc2')
    ax.legend(loc='best')
    plt.show()
    return

# useless?
def select_correspondence():
    print('selecting new crosspondence...')
    global g_crt_match1
    g_crt_match1 = (0, 0)
    global g_crt_match2
    g_crt_match2 = (0, 0)
    global g_vis_img1
    g_vis_img1 = np.copy(g_color_img1)
    global g_vis_img2
    g_vis_img2 = np.copy(g_color_img2)
    global g_rand_color
    g_rand_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

# save
def save_correspondence():
    if g_crt_match1 == (0, 0) or g_crt_match2 == (0, 0):
        print('wrong operation.')
    else:
        global g_corres_l1
        global g_corres_l2
        g_corres_l1.append(g_crt_match1)
        g_corres_l2.append(g_crt_match2)
        print('the crosspondence saved.')

# process before solve Rt
def process_current_corres():
    global g_p3d_l1
    global g_p3d_l2
    g_p3d_l1 = get_3d_points(g_corres_l1, g_depth_img1)
    g_p3d_l2 = get_3d_points(g_corres_l2, g_depth_img2)
    print('#correspondences', len(g_p3d_l1))

# solve Rt
def solve_Rt():
    if len(g_p3d_l1) < 4:
        print('correspondences not enough.')
        return
    global g_R, g_t
    kab = Kabsch(g_p3d_l1, g_p3d_l2)
    g_R, g_t = kab.solve_R_t() # solve
    Rt = np.zeros((4, 4))
    Rt[0:3, 0:3] = g_R
    Rt[0:3, 3] = g_t
    Rt[3, 3] = 1.
    #lp3d1 = np.array(generate_pc(g_depth_img1))
    #lp3d2 = np.array(generate_pc(g_depth_img2, Rt=Rt))
    #draw_merged_pc(lp3d1, lp3d2)
    lp3d1 = np.array(generate_pc_color(g_depth_img1, g_color_img1))
    lp3d2 = np.array(generate_pc_color(g_depth_img2, g_color_img2, Rt=Rt))
    draw_merged_pc_color(lp3d1, lp3d2)

# reset
def clean_correspondences():
    global g_crt_match1, g_crt_match2, g_corres_l1, g_corres_l2, g_p3d_l1, g_p3d_l2
    g_crt_match1 = (0, 0)
    g_crt_match2 = (0, 0)
    g_corres_l1 = []
    g_corres_l2 = []
    g_p3d_l1 = []
    g_p3d_l2 = []

# get 3d from 2d
def get_3d_points(l_p2d, img_depth):
    l_p3d = []
    for p2d in l_p2d:
        c = p2d[0]
        r = p2d[1]
        depth = img_depth[r,c]
        z = depth / g_camera_factor
        x = float(c-g_cx) * z / g_fx
        y = float(r-g_cy) * z / g_fy
        l_p3d.append((x, y, z))
    return l_p3d

# to be update
def video_loop():
    imgtk_1 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(g_vis_img1, cv2.COLOR_BGR2RGBA)))
    img_panel_1.imgtk = imgtk_1
    img_panel_1.config(image=imgtk_1)
    imgtk_2 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(g_vis_img2, cv2.COLOR_BGR2RGBA)))
    img_panel_2.imgtk = imgtk_2
    img_panel_2.config(image=imgtk_2)
    root.after(1, video_loop)

# check depth validness
def depth_valid(x, y, img_depth):
    depth = img_depth[y,x]
    if depth==0 or depth==65535:
        return False
    return True

# mouse event
def callback_1(event):
    if not depth_valid(event.x, event.y, g_depth_img1):
        print('depth invalid.')
        return
    # save pixel coord
    global g_crt_match1
    g_crt_match1 = (event.x, event.y)
    # show
    global g_vis_img1
    g_vis_img1 = np.copy(g_color_img1)
    cv2.circle(g_vis_img1, (event.x, event.y), 6, g_rand_color, -1)

# mouse event
def callback_2(event):
    if not depth_valid(event.x, event.y, g_depth_img2):
        print('depth invalid.')
        return
    # save pixel coord
    global g_crt_match2
    g_crt_match2 = (event.x, event.y)
    # show
    global g_vis_img2
    g_vis_img2 = np.copy(g_color_img2)
    cv2.circle(g_vis_img2, (event.x, event.y), 6, g_rand_color, -1)


if __name__ == "__main__":

    root = Tk()
    root.title("opencv + tkinter")

    # image panels
    img_panel_1 = Label(root)  
    img_panel_1.bind("<Button-1>", callback_1)
    img_panel_1.pack(side=LEFT, padx=10, pady=10)
    img_panel_2 = Label(root)  
    img_panel_2.bind("<Button-1>", callback_2)
    img_panel_2.pack(side=RIGHT, padx=10, pady=10)
    root.config(cursor="arrow")

    # btns
    confirm_btn = Button(root, text=    "save correspondence", command=save_correspondence)
    confirm_btn.pack(expand=False, padx=10, pady=10)
    select_btn = Button(root, text=     "new correspondence", command=select_correspondence)
    select_btn.pack(expand=False, padx=10, pady=10)
    list_btn = Button(root, text=       "process current corres", command=process_current_corres)
    list_btn.pack(expand=False, padx=10, pady=10)
    solve_btn = Button(root, text=      "solve Rt", command=solve_Rt)
    solve_btn.pack(expand=False, padx=10, pady=10)
    clean_btn = Button(root, text=      "clean corres", command=clean_correspondences)
    clean_btn.pack(expand=False, padx=10, pady=10)

    video_loop()
    root.mainloop()
    cv2.destroyAllWindows()
