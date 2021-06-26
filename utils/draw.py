import matplotlib.pyplot as plt


def draw_loss(train_loss, eval_loss, name):
    plt.plot(train_loss, label='train_loss')
    plt.plot(eval_loss, label='eval_loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.legend(['train','eval'], loc='upper right')
    plt.savefig('./{}.png'.format(name))