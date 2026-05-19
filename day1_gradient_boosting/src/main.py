from residual_learning import run_residual_demo

from boosting_loop import boosting_loop

from experiments import (
    learning_rate_experiment,
    tree_experiment
)

from visualization import plot_learning_curve


def main():

    # Residual demo
    run_residual_demo()

    # Manual boosting loop
    boosting_loop()

    # Learning rate experiments
    learning_rate_experiment()

    # Tree count experiments
    tree_experiment()

    # Visualization
    plot_learning_curve()


if __name__ == "__main__":
    main()