from pathlib import Path
import torch
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

def compare_results(y, pred):
    print( torch.cat(y, pred))

def list_models(model_idx=None):
    models_path = Path(__file__).parent / "models"
    models = sorted(models_path.glob("MODEL_*"))

    if model_idx is not None:
        return models[model_idx]

    for i, model_path in enumerate(models):
        print(i, model_path)


if __name__ == "__main__":
    from sys import argv 
    if len(argv) < 2:
        list_models()
        exit()
    model_path = list_models(int(argv[1]))    

    import inference as inf
    import train_gat2
    train_gat2.DATA_DIR = train_gat2.DATA_DIR.parent / argv[3]
    data_train, data_val = train_gat2.get_data()
    model = inf.load_model(model_path, train_gat2.model)
    inf.inference(data_val[int(argv[2])], model)

    
