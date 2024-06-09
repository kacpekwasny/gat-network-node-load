import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

def comparison_plot(baseline_loss_traj):
    '''
    Plot the trajectory of a model's training from a dataframe that has train/val loss at each epoch. 
    '''
    plt.yscale('log') 
    # list(zip(*l)) unzips a list of tuples l into a list of lists
    # in our case, the first elem is epoch, second is train loss, third is val loss
    epoch, _, baseline_val = tuple(zip(*baseline_loss_traj))
    # _, _, model_val = tuple(zip(*model_loss_traj))
    plt.plot(epoch, baseline_val, '-r', label='baseline')
    # plt.plot(epoch, model_val, '-b', label='model')
    plt.legend(loc='upper right', title='model type')
    plt.yscale('log')
    plt.ylabel('log val MSE')
    plt.xlabel('epoch')
    plt.title(f'Comparing baseline and model with edge features')
    plt.show()
    plt.clf()


def draw_results(edge_list, pred, Y) -> None:
    g = nx.from_edgelist(edge_list)
    color_map_pred = []
    color_map_y = []
    for p, y in zip(pred, Y):
        color_map_pred.append(p.numpy()[0])
        color_map_y.append(y.numpy())
    nx.draw(g, node_color=color_map_y)
    plt.savefig('pred.png')

    nx.draw(g, node_color=color_map_pred)
    plt.savefig('pred.png')

    pass
