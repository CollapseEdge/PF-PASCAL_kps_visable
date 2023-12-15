import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import math, cv2
from tqdm import tqdm
import numpy as np


def ProcessCoordinate(kpsX):
    return [float(x+512) for x in kpsX]


def DrawPoint(img, picName, XA, YA, XB, YB, kpsX, kpsY, thres1):
    colors_list = ['#9CFD86', '#5CE217', '#BC935C', '#E6FB2A', '#EA254F', '#E127D9', '#7968B6', '#663D02', '#90EFA0', '#34B02E', '#8098AF', '#219F80', '#891D0B', '#3515C3', '#E91134', '#37D4C2', '#D5CA4F', '#C93BC9', '#3B0CCE', '#F7986C', '#51BA78', '#C09FF7', '#F10800', '#9B3F26', '#EE2790', '#BB6F5A', '#4AECB5', '#BA9096', '#E48004', '#C2FB5F', '#507F66', '#476718', '#A09447', '#BF523C', '#2A5ACB', '#E60E8C', '#58E7C1', '#E9FBF5']
    kpsX = ProcessCoordinate(kpsX)  # 处理坐标
    G = nx.Graph()
    G1 = nx.Graph()
    G2 = nx.Graph()
    G3 = nx.Graph()
    fig, ax = plt.subplots()
    fig.set_size_inches(5.13, 2.56)
    ax.imshow(img)
    for i in range(len(XA)):
        color = colors_list[i]
        source_points = (int(float(XA[i])), int(float(YA[i])))
        target_points = (int(float(XB[i])), int(float(YB[i])))
        kps_points = (int(float(kpsX[i])), int(float(kpsY[i])))
        G1.add_node(source_points, color=color)
        G.add_node(target_points, color=color)
        G2.add_node(kps_points, color=color)
        l2dist = math.sqrt(
            (float(kpsX[i]) - float(XA[i])) * (float(kpsX[i]) - float(XA[i])) + (float(kpsY[i]) - float(YA[i])) * (
                        float(kpsY[i]) - float(YA[i])))
        if l2dist <= thres1:
            correct_pts = True
        else:
            correct_pts = False
        if correct_pts == True:
            #G3.add_edge(kps_points, target_points, color='g')
            G3.add_edge(kps_points, source_points, color='g')
        else:
            #G3.add_edge(kps_points, target_points, color='r')
            G3.add_edge(kps_points, source_points, color='r')
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    pos1 = {node: (node[0], node[1]) for node in G1.nodes()}
    pos2 = {node: (node[0], node[1]) for node in G2.nodes()}
    pos3 = {node: (node[0], node[1]) for node in G3.nodes()}
    
    nx.draw_networkx_nodes(G, pos, node_size=20, node_shape='o', node_color=[color for color in [G.nodes[node]['color'] for node in G.nodes()]], edgecolors='black', ax=ax)
    nx.draw_networkx_nodes(G1, pos1, node_size=20, node_shape='d', node_color=[color for color in [G1.nodes[node]['color'] for node in G1.nodes()]], edgecolors='black', ax=ax)
    nx.draw_networkx_nodes(G2, pos2, node_size=20, node_shape='o', node_color=[color for color in [G2.nodes[node]['color'] for node in G2.nodes()]],edgecolors='grey', ax=ax)
    nx.draw_networkx_edges(G3, pos3, width=2, alpha=1, edge_color=[color for color in [G3[u][v]['color'] for u, v in G3.edges()]], arrows=False, ax=ax)
    plt.xlim(0, 1024 - 1)
    plt.ylim(512 - 1, 0)
    png_name = './VAT/' + str(picName) + '.png'
    plt.axis('off')
    plt.savefig(png_name, format="png", bbox_inches='tight', pad_inches=0,dpi = 600)
    plt.close()
    G.clear()
    G1.clear()
    G2.clear()
    G3.clear()
    return 0

def getpic_name(string):
    pic_full_name = string.split('/')[-1]
    pic_name = pic_full_name.split('.')[0]
    return pic_name

def concat_pic_name(pic_s,pic_t):
    return pic_s + '-' + pic_t

def read_and_process_csv(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 去除[]和-1
    df['x'] = df['x'].apply(lambda x: eval(x) if pd.notnull(x) else [])
    df['y'] = df['y'].apply(lambda y: eval(y) if pd.notnull(y) else [])

    # 去除-1
    df['x'] = df['x'].apply(lambda x: [value for value in x if value != -1])
    df['y'] = df['y'].apply(lambda y: [value for value in y if value != -1])

    return df


def main():
    df = pd.read_csv('./test_pairs.csv')
    df_1 = read_and_process_csv('./coordinates_2.csv')
    #print(df_1['x'][0])
    i = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc='Processing images'):
        source_path = './' + row['source_image']
        target_path = './' + row['target_image']
        source_name = getpic_name(source_path)
        target_name = getpic_name(target_path)
        save_name = concat_pic_name(source_name,target_name)

        # 读取源图像和目标图像
        source_img = cv2.imread(source_path)
        target_img = cv2.imread(target_path)
        if source_img is None or target_img is None:
            raise Exception(f"Failed to read one or both images. Check file paths and integrity.")
        
        # 将图像从BGR格式转换为RGB格式
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # Get the original dimensions of the images
        original_height_s, original_width_s, _ = source_img.shape
        original_height_t, original_width_t, _ = target_img.shape

        # Resize each image to 512x512
        source_img = cv2.resize(source_img, (512, 512))
        target_img = cv2.resize(target_img, (512, 512))
        # 将两个图像拼接起来
        result_img = np.concatenate((target_img, source_img), axis=1)
        #result_img = np.concatenate((source_img, target_img), axis=1)
        # 获取坐标点
        x_coords_source = list(map(float, row['XA'].split(';')))
        y_coords_source = list(map(float, row['YA'].split(';')))
        x_coords_target = list(map(float, row['XB'].split(';')))
        y_coords_target = list(map(float, row['YB'].split(';')))

        # 缩放坐标到新的图像大小 (512x512)
        x_coords_source_scaled = [int(x * (512 / original_width_s)) + 512 for x in x_coords_source]
        y_coords_source_scaled = [int(y * (512 / original_height_s)) for y in y_coords_source]

        x_coords_target_scaled = [int(x * (512 / original_width_t)) for x in x_coords_target]
        y_coords_target_scaled = [int(y * (512 / original_height_t)) for y in y_coords_target]

        x_coords_processed = [int(x * (512 / 512)) + 0 if x != -1 else -1 for x in df_1['x'][i]]
        y_coords_processed = [int(y * (512 / 512)) if y != -1 else -1 for y in df_1['y'][i]]
        DrawPoint(img = result_img, picName = save_name, XA=x_coords_source_scaled, YA=y_coords_source_scaled,XB=x_coords_target_scaled,YB=y_coords_target_scaled,kpsX=x_coords_processed,kpsY=y_coords_processed,thres1=51.2)
        i+=1
        

if __name__ == "__main__":
    main()