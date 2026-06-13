# Basics
import random
from tqdm import tqdm
from tensorflow.keras.backend import clear_session

# Custom
from network import Network
from genetic_tuner import GeneticTuner

class HPTuner():

    """
    Neural Network hyperparameter tuning with differnt techniques.

    Attributes:
        - model_type (str):
            Type of Neural Network to implement.
        - hp_space (dict):
            Whole hyperparameter space.
        - n_models (int):
            Number of models to generate and evaluate for each trial.
        - n_epochs (int):
            Number of training epochs to use in each trial.
        - best_val_loss (float):
            Validation loss of the best-performing hyperparameters.
        - best_hp (dict):
           Best-performing hyperparameters.
        - best_history (dict):
            Training history of the best-performing hyperparameters.

    Methods:
        - random_search:
            Performs hyperparameter tuning with a Random Search.
        - genetic_algorithm:
            Performs hyperparameter tuning with a Genetic Algorithm.
    """


    def __init__(self, model_type, hp_space, n_models, n_epochs):

        """
        Class constructor: takes as input all class attributes and generates a
        HPTuner object.
        """

        self.model_type = model_type
        self.hp_space = hp_space
        self.n_models = n_models
        self.n_epochs = n_epochs


    def random_search(self, x_train, y_train, x_val, y_val):

        """
        Performs hyperparameter tuning with a Random Search: a fixed number of
        random hyperparameters is selected from the hyperparameter space and tested.

        Only the best-performing hyperparameters are considered as result.

        The performance of the hyperparameters is given by the validation loss
        (the lower the better).

        The number of random hyperparameters to generate is specified in the
        constructor as "n_models".

        Parameters:
            - x_train, x_val (np.ndarray):
                Train and validation samples.
            - y_train, y_val (np.ndarray):
                Train and validation labels.

        Returns:
            - best_hp (dict):
                Best-performing hyperparameters.
        """

        res = []
        for _ in tqdm(range(self.n_models), desc='Random Search: '):

            clear_session()

            random_hp = {k: random.choice(self.hp_space[k]) for k in self.hp_space.keys()}
            net = Network(self.model_type, random_hp)
            net.build_model()
            model = net.model

            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=self.n_epochs,
                batch_size=net.hp['batch_size'],
                callbacks=net.callbacks,
                verbose=0
            ).history

            # val_loss, _ = model.evaluate(x_val, y_val, verbose=0)
            val_loss = history['val_loss'][-1]
            res.append((val_loss, random_hp, history))

            # delete model
            del model

        res.sort(key=lambda x: x[0])

        self.best_val_loss, self.best_hp, self.best_history = res[0] # Take track of the best results

        print(f'Best result: {self.best_val_loss}')

        return self.best_hp


    def genetic_algorithm(self, n_gen, selection_perc, second_chance_prob, mutation_prob,
                          x_train, y_train, x_val, y_val):

        """
        Performs hyperparameter tuning with a Genetic Algorithm: populations of
        hyperparameters are generated, tested and updated according to the
        "natural selection" principle (the best-performing hyperparameters are
        kept to generate the next population).

        Only the hyperparameters that show the best performance after a fixed
        number of generations are considered as result.

        The performance of the hyperparameters is given by the validation loss
        (the lower the better).

        The size of each population is is specified in the constructor as
        "n_models".

        Parameters:
            - n_gen (int):
                Number of generations to consider.
            - selection_perc, second_chance_prob, mutation_prob (float):
                Genetic Algorithm parameters used to tune the entire process
                (how many best-performing hyperparameters to consider, how many
                bad-performing hyperparameters to consider, mutation probability).
            - x_train, x_val (np.ndarray):
                Train and validation samples.
            - y_train, y_val (np.ndarray):
                Train and validation labels.

        Returns:
            - best_hp (dict):
                Best-performing hyperparameters.
        """

        gt = GeneticTuner(
            model_type=self.model_type,
            hp_space=self.hp_space,
            pop_size=self.n_models,
            selection_perc=selection_perc,
            second_chance_prob=second_chance_prob,
            mutation_prob=mutation_prob
        )

        pop = gt.populate()

        for gen in tqdm(range(n_gen)):
            evaluation = gt.evaluate(
                pop=pop,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                n_epochs=self.n_epochs
            )

            # print(f'Top 5: {[val_loss for val_loss, _, _ in evaluation[:5]]}')

            parents = gt.select(evaluation)

            if gen != n_gen-1:
                pop = gt.evolve(parents)


        self.best_val_loss, self.best_hp, self.best_history = evaluation[0]

        return self.best_hp
