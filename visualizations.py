from matplotlib import pyplot as plt

def plot_coolwarm_side_by_side(features1, targets1, features2, targets2):
    ax = plt.subplot(1, 2, 1)
    # ax.set_title("")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(features1["longitude"],
                features1["latitude"],
                cmap="coolwarm",
                c = targets1["median_house_value"] / targets1["median_house_value"].max())

    ax = plt.subplot(1,2,2)
    # ax.set_title("")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(features2["longitude"],
                features2["latitude"],
                cmap="coolwarm",
                c = targets2["median_house_value"] / targets2["median_house_value"].max())
    _ = plt.show()

def plot_rmse_side_by_side(training_rmse, validation_rmse):
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label = "training")
  plt.plot(validation_rmse, label = "validation")
  plt.legend()
  plt.show()