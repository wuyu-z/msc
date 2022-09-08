import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.transform import resize

img_size=224
downsample=2
pad_pixels=50
legend_margin=1000
dpi=100

def plot_and_save(wsi,colors,sort,meta_field):
    from matplotlib.lines import Line2D
    #image_clusters, counts = np.unique(slide_clusters, return_counts=True)
    custom_lines = [Line2D([0], [0], color=colors[sort.index(i)], lw=2.5) for i in sort]
    names_lines = ['Cluster %s' % str(i) for i in sort]
    height, width, _ = wsi.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    legend = ax.legend(custom_lines, names_lines, title='Leiden Clusters', loc='upper right', prop={'size': 40})
    legend.get_title().set_fontsize('48')
    legend.get_frame().set_linewidth(1)
    ax.imshow(wsi)
    fig.savefig('./output/%s.png'% meta_field,dpi=dpi,transparent=True)
    plt.show()

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def get_x_y(tile_info):
    if '.' in str(tile_info):
        string = tile_info.split('.')[0]
    else:
        string = str(tile_info)
    x, y   = string.split('_')
    return int(x),int(y)

def save_txt(cluster,meta):
    luad=[35, 22, 36, 28, 31, 11]
    lusc=[45,5]
    survive= [31, 1, 37, 0, 16, 8, 5]
    death= [7, 3, 14, 20, 41, 39, 15]
    if meta == 'lungsubtype':
        with open('./output/result.txt','w') as output:
            output.write('-'*50)
            output.write('\n')
            output.write("Most dominant cluster of %s is %d\n" %(meta,cluster))
            subtype=''
            if cluster in luad:
                subtype='luad'
            elif cluster in lusc:
                subtype='lusc'
            else:
                subtype='neutral'
            output.write('Subtype favour %s \n'% (subtype))
            output.close()
    elif meta == 'survival':
        with open('./output/result.txt', 'a') as output:
            output.write('-' * 50)
            output.write('\n')
            output.write("Most dominant cluster of %s is %d\n" % (meta, cluster))
            survivetype=''
            if cluster in survive:
                survivetype = 'survive'
            elif cluster in death:
                survivetype = 'death'
            else:
                survivetype = 'neutral'
            output.write('Survivability favour %s \n' % survivetype)
            output.close()

def plot_wsi(csv,h5,meta_field):
    max_x=0
    max_y=0
    cluster=pd.read_csv(csv)
    color_cluster,color_count=np.unique(cluster['leiden_2.0'].values.astype(int),return_counts=True)
    color_sort=np.argsort(-color_count)
    color_cluster_sort=color_cluster[color_sort].tolist()
    colors = sns.color_palette('tab20', len(color_cluster_sort))
    dominant=color_cluster_sort[0]
    save_txt(dominant,meta_field)
    for i in cluster['combined_tiles']:
        y_current,x_current=get_x_y(i)
        max_x=max(max_x,x_current)
        max_y=max(max_y,y_current)
    max_x+=1
    max_y+=1
    wsi_x = int(max_x * img_size // downsample)
    wsi_y = int(max_y * img_size // downsample)
    wsi_c = np.ones((wsi_x, wsi_y, 3), dtype=np.uint8)*255

    with h5py.File(h5,'r') as h5:
        for i in range(len(h5['combined_tiles'])):
            combined_tile=h5['combined_tiles'][i].astype(str)
            y_i, x_i =get_x_y(combined_tile)
            x_i *= img_size // downsample
            y_i *= img_size // downsample
            cluster_id=cluster[cluster['combined_tiles']==combined_tile]['leiden_2.0'].item()

        # for index, row in cluster.iterrows():
        #     y_i, x_i = get_x_y(row['combined_tiles'])
        #     x_i *= img_size // downsample
        #     y_i *= img_size // downsample
        #     i=0
        #     for j in range(len(h5['combined_tiles'])):
        #         if h5['combined_tiles'][j].astype(str)==row['combined_tiles']:
        #             i=j
        #             break

            tile_img = h5['combined_img'][i]
            tile_img = np.array(resize(tile_img, (tile_img.shape[0] // downsample, tile_img.shape[1] // downsample), anti_aliasing=True),dtype=float)
            tile_img = (tile_img * 255).astype(np.uint8)
            color = colors[color_cluster_sort.index(cluster_id)]
            mask = np.ones((img_size // downsample, img_size // downsample))
            wsi_c[x_i:x_i + (img_size // downsample), y_i:y_i + (img_size // downsample), :] = apply_mask(tile_img, mask, color,alpha=0.5)

        wsi_padded = np.pad(wsi_c[:, :, 0], ((pad_pixels, pad_pixels), (pad_pixels, pad_pixels + legend_margin)), 'maximum')

        wsi_c_padded_total = np.zeros(list(wsi_padded.shape) + [3])
        wsi_c_padded_total[:, :, 0] = wsi_padded
        wsi_c_padded_total[:, :, 1] = np.pad(wsi_c[:, :, 1],((pad_pixels, pad_pixels), (pad_pixels, pad_pixels + legend_margin)),'maximum')
        wsi_c_padded_total[:, :, 2] = np.pad(wsi_c[:, :, 2],((pad_pixels, pad_pixels), (pad_pixels, pad_pixels + legend_margin)),'maximum')
    plot_and_save(wsi_c,colors,color_cluster_sort,meta_field)

    '''fig= plt.figure()
        ax = fig.add_subplot(111)
        tile_img=h5['combined_img'][1]
        tile_img = np.array(resize(tile_img, (tile_img.shape[0] // 2, tile_img.shape[1] // 2), anti_aliasing=True),dtype=float)
        tile_img = (tile_img * 255).astype(np.uint8)
        ax.imshow(tile_img)'''
    #plt.show()