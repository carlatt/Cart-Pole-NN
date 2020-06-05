import matplotlib.pyplot as plt


def plot_results(realcartpos, predcartpos, realcartvel, predcartvel, realtheta, predtheta, realomega, predomega):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    ax1.plot(realcartpos, 'b', label='Real pos')
    ax1.plot(predcartpos, 'r:', label='Pred pos')
    ax1.legend(loc="upper right")
    ax2.plot(realcartvel, 'b', label='Real vel')
    ax2.plot(predcartvel, 'r:', label='Pred vel')
    ax2.legend(loc="upper right")
    ax3.plot(realtheta, 'b', label='Real theta')
    ax3.plot(predtheta, 'r:', label='Pred theta')
    ax3.legend(loc="upper right")
    ax4.plot(realomega, 'b', label='Real omega')
    ax4.plot(predomega, 'r:', label='Pred omega')
    ax4.legend(loc="upper right")

    plt.show()


def plot_errors(Y, pred):
    error = Y - pred
    error_pos = [item[0] for item in error]
    error_vel = [item[1] for item in error]
    error_theta = [item[2] for item in error]
    error_omega = [item[3] for item in error]

    fig, ax = plt.subplots()
    ax.plot(error_pos, 'r', label='error_pos')
    ax.plot(error_vel, 'b', label='error_vel')
    ax.plot(error_theta, 'k', label='error_theta')
    ax.plot(error_omega, 'g', label='error_omega')

    legend = ax.legend(loc='upper right')

    plt.show()


def plot_NRMSE(NRMSEs):
    plt.scatter(range(1, 1 + len(NRMSEs)), NRMSEs, label="NRMSE for each model", color='r', s=100)
    plt.legend(loc='upper left')
    plt.xticks([1, 2, 3, 4, 5])
    plt.show()
