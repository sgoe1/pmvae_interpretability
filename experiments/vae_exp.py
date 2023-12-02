import anndata
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from captum.attr import GradientShap, IntegratedGradients, Saliency
from lfxai.explanations.features import attribute_auxiliary
from lfxai.models.pretext import Identity, Mask, RandomNoise
from lfxai.explanations.examples import (
    InfluenceFunctions,
    NearestNeighbours,
    SimplEx,
    TracIn,
)
from lfxai.utils.metrics import (
    compute_metrics,
    cos_saliency,
    count_activated_neurons,
    entropy_saliency,
    pearson_saliency,
    similarity_rates,
    spearman_saliency,
)
from lfxai.utils.visualize import (
    plot_pretext_saliencies_lineplot,
    plot_pretext_top_example_lineplot,
)
from lfxai.utils.feature_attribution import generate_masks

import random

from sklearn.model_selection import train_test_split

from scipy.stats import spearmanr


from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from lfxai.models.vae import pmVAEModel, LogisticRegression


def parse_gmt(path, symbols=None, min_genes=10):
    lut = dict()
    for line in open(path, "r"):
        key, _, *genes = line.strip().split()
        if symbols is not None:
            genes = symbols.intersection(genes).tolist()
        if len(genes) < min_genes:
            continue
        lut[key] = genes

    return lut


def load_annotations(gmt, genes, min_genes=10):
    genesets = parse_gmt(gmt, genes, min_genes)
    annotations = pd.DataFrame(False, index=genes, columns=genesets.keys())
    for key, genes in genesets.items():
        annotations.loc[genes, key] = True

    return annotations


class RNASeqData(Dataset):
    def __init__(self, X, c=None, y=None, transform=None):
        self.X = X
        self.y = y
        self.c = c
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        sample = self.X[index, :]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.y is not None and self.c is not None:
            return sample, self.y[index], self.c[index]
        if self.y is None and self.c is not None:
            return sample, self.c[index]
        else:
            return sample


def load_data(filename, device):
    """
    Loads dataset and returns it in all forms needed for downstream use.
    """
    data_dir = Path.cwd() / "data/pause_sc_datasets"
    data = anndata.read(f"{data_dir}/{filename}.h5ad")
    symbols = data.var_names

    data.varm["I"] = load_annotations(
        f"{data_dir}/c2.cp.reactome.v7.4.symbols.gmt", symbols, min_genes=13
    ).values
    data.uns["terms"] = list(
        load_annotations(
            f"{data_dir}/c2.cp.reactome.v7.4.symbols.gmt", symbols, min_genes=13
        ).columns
    )

    rand_seed = 939
    train_data, test_data = train_test_split(
        data, test_size=0.25, shuffle=True, random_state=rand_seed
    )
    tr_data, val_data = train_test_split(
        train_data, test_size=0.25, shuffle=True, random_state=rand_seed
    )

    filename_to_fields = {
        "kang_PBMC": ["condition", "stimulated", 979],
        "norman_k562_crispr": ["gene_program", "Ctrl"],
        # "datlinger": ["condition", "stimulated"],
        # "mcfarland": ["TP53_mutation_status", "Wild Type"],
        "haber_intestinal_epithelial": ["condition", "Control", 2000],
    }

    group_1 = filename_to_fields[filename][0]
    group_2 = filename_to_fields[filename][1]
    data_dim = filename_to_fields[filename][2]

    y_tr = tr_data.obs[group_1]
    y_val = val_data.obs[group_1]
    y_test = test_data.obs[group_1]

    train_labels = (y_tr == group_2).values
    train_labels = torch.tensor(train_labels, dtype=torch.float32)

    val_labels = (y_val == group_2).values
    val_labels = torch.tensor(val_labels, dtype=torch.float32)

    test_labels = (y_test == group_2).values
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    tr_ds = RNASeqData(torch.tensor(tr_data.X))
    val_ds = RNASeqData(torch.tensor(val_data.X))
    test_ds = RNASeqData(torch.tensor(test_data.X))
    # tr_ds = torch.tensor(RNASeqData(tr_data))
    # val_ds = torch.tensor(RNASeqData(val_data))
    # test_ds = torch.tensor(RNASeqData(test_data))

    X_train = torch.tensor(tr_data.X).to(device)
    y_train = train_labels.to(device)
    X_val = torch.tensor(val_data.X).to(device)
    y_val = val_labels.to(device)
    X_test = torch.tensor(test_data.X).to(device)
    y_test = test_labels.to(device)

    membership_mask = (
        load_annotations(
            f"{data_dir}/c2.cp.reactome.v7.4.symbols.gmt", symbols, min_genes=13
        )
        .astype(bool)
        .T
    )

    return (
        tr_ds,
        val_ds,
        test_ds,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        data_dim,
        membership_mask,
    )


def pretext_task_sensitivity(filename, device=torch.device("cuda:0"), n_epochs=2):
    (
        tr_ds,
        val_ds,
        test_ds,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        data_dim,
        membership_mask,
    ) = load_data(filename, device)

    checkpoint_dir = Path.cwd() / "results/vae"
    checkpoint_path = f"{checkpoint_dir}/{filename}_baseModel.pkl"

    save_dir = Path.cwd() / "results/vae/pretext"

    subtrain_size = 2500
    idx_subtrain = random.sample(range(0, subtrain_size), subtrain_size)

    # train
    batch_size = 256
    mse_loss = torch.nn.MSELoss()

    n_runs = 1
    pretext_list = [Identity(), RandomNoise(noise_level=0.3), Mask(mask_proportion=0.2)]
    n_tasks = len(pretext_list) + 1
    feature_pearson = np.zeros((n_runs, n_tasks, n_tasks))
    feature_spearman = np.zeros((n_runs, n_tasks, n_tasks))
    example_pearson = np.zeros((n_runs, n_tasks, n_tasks))
    example_spearman = np.zeros((n_runs, n_tasks, n_tasks))

    feature_importance = []
    example_importance = []
    for pretext in pretext_list:
        # Create and fit an autoencoder for the pretext task
        name = f"{str(pretext)}-ae_run1"

        print(f"Training {name}")
        basePMVAE = pmVAEModel(
            membership_mask.values,
            [12],
            4,
            beta=1e-05,
            terms=membership_mask.index,
            add_auxiliary_module=False,
            use_gpu=True,
            input_pert=pretext,
        )
        basePMVAE.train(
            tr_ds,
            val_ds,
            checkpoint_path=checkpoint_path,
            max_epochs=n_epochs,
            pathway_dropout=True,
            batch_size=batch_size,
        )

        gradshap = GradientShap(basePMVAE.encoder)
        test_dataloader = torch.utils.data.DataLoader(
            test_ds, batch_size=256, shuffle=False, num_workers=2
        )
        print("Computing feature importance")
        baseline = torch.randn(batch_size, data_dim).to(device)
        feature_importance.append(
            np.abs(
                np.expand_dims(
                    attribute_auxiliary(
                        basePMVAE.encoder, test_dataloader, device, gradshap, baseline
                    ),
                    0,
                )
            )
        )
        # Compute example importance
        print("Computing example importance")
        dknn = NearestNeighbours(
            basePMVAE.encoder,
            loss_f=mse_loss,
            X_train=X_train,
        )
        example_importance.append(
            np.expand_dims(dknn.attribute(X_test, idx_subtrain).cpu().numpy(), 0)
        )

    # Train classifier
    classifier = LogisticRegression(data_dim, 1).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)

    classifier.fit(X_train, y_train, X_val, y_val, criterion, optimizer, epochs=50)
    gradshap = GradientShap(classifier)
    test_dataloader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=2
    )
    baseline = torch.randn(batch_size, data_dim).to(device)

    feature_importance.append(
        np.abs(
            np.expand_dims(
                attribute_auxiliary(
                    classifier, test_dataloader, device, gradshap, baseline
                ),
                0,
            )
        )
    )

    dknn = NearestNeighbours(classifier, loss_f=mse_loss, X_train=X_train)
    example_importance.append(
        np.expand_dims(dknn.attribute(X_test, idx_subtrain).cpu().numpy(), 0)
    )

    # Compute correlation between the saliency of different pretext tasks
    run = 0
    feature_importance = np.concatenate(feature_importance)
    feature_pearson[run] = np.corrcoef(feature_importance.reshape((n_tasks, -1)))
    feature_spearman[run] = spearmanr(
        feature_importance.reshape((n_tasks, -1)), axis=1
    )[0]
    example_importance = np.concatenate(example_importance)
    example_pearson[run] = np.corrcoef(example_importance.reshape((n_tasks, -1)))
    example_spearman[run] = spearmanr(
        example_importance.reshape((n_tasks, -1)), axis=1
    )[0]

    print(
        f"Run {run} complete \n Feature Pearson \n {np.round(feature_pearson[run], decimals=2)}"
        f"\n Feature Spearman \n {np.round(feature_spearman[run], decimals=2)}"
        f"\n Example Pearson \n {np.round(example_pearson[run], decimals=2)}"
        f"\n Example Spearman \n {np.round(example_spearman[run], decimals=2)}"
    )

    # Plot a couple of examples
    headers = [str(pretext) for pretext in pretext_list] + ["Classification"]
    n_plots = 10
    idx_plot = random.sample(range(0, n_plots), n_plots)
    # test_images_to_plot = [X_test[i][0].numpy() for i in idx_plot]
    # train_images_to_plot = [
    #     X_train[i][0].numpy() for i in idx_subtrain
    # ]
    test_images_to_plot = [X_test[i].detach().cpu().numpy() for i in idx_plot]
    train_images_to_plot = [X_train[i].detach().cpu().numpy() for i in idx_subtrain]

    fig_features = plot_pretext_saliencies_lineplot(
        test_images_to_plot, feature_importance[:, idx_plot, :], headers
    )
    fig_features.savefig(save_dir / f"saliency_maps_run{run}.png")
    plt.close(fig_features)

    fig_examples = plot_pretext_top_example_lineplot(
        train_images_to_plot,
        test_images_to_plot,
        example_importance[:, idx_plot, :],
        headers,
    )
    fig_examples.savefig(save_dir / f"top_examples_run{run}.png")
    plt.close(fig_features)


def consistency_feature_importance(
    filename: str,
    random_seed: int = 1,
    batch_size: int = 256,
    dim_latent: int = 4,
    n_epochs: int = 2,
    device: torch.device = torch.device("cuda:0"),
) -> None:
    # Initialize seed and device
    # torch.random.manual_seed(random_seed)
    pert_percentages = [5, 10, 20, 50, 80, 100]

    (
        tr_ds,
        val_ds,
        test_ds,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        data_dim,
        membership_mask,
    ) = load_data(filename, device)

    checkpoint_dir = Path.cwd() / "results/vae"
    checkpoint_path = f"{checkpoint_dir}/{filename}_baseModel.pkl"

    save_dir = Path.cwd() / "results/vae/consistency_features"

    train_loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    # Initialize encoder, decoder and autoencoder wrapper
    pert = RandomNoise()
    basePMVAE = pmVAEModel(
        membership_mask.values,
        [12],
        dim_latent,
        beta=1e-05,
        terms=membership_mask.index,
        add_auxiliary_module=False,
        use_gpu=True,
    )
    basePMVAE.train(
        tr_ds,
        val_ds,
        checkpoint_path=checkpoint_path,
        max_epochs=n_epochs,
        pathway_dropout=True,
        batch_size=batch_size,
    )

    attr_methods = {
        "Gradient Shap": GradientShap,
        "Integrated Gradients": IntegratedGradients,
        "Saliency": Saliency,
        "Random": None,
    }
    results_data = []
    # baseline = torch.zeros(1, 1, data_dim).to(device)
    baseline = torch.randn(1, data_dim).to(device)
    for method_name in attr_methods:
        print(f"Computing feature importance with {method_name}")
        results_data.append([method_name, 0, 0])
        attr_method = attr_methods[method_name]
        if attr_method is not None:
            attr = attribute_auxiliary(
                basePMVAE.encoder,
                test_loader,
                device,
                attr_method(basePMVAE.encoder),
                baseline,
            )
        else:
            # np.random.seed(random_seed)
            attr = np.random.randn(len(test_ds), data_dim)

        for pert_percentage in pert_percentages:
            print(f"Perturbing {pert_percentage}% of the features with {method_name}")
            mask_size = int(pert_percentage * data_dim / 100)
            masks = generate_masks(attr, mask_size)
            for batch_id, images in enumerate(test_loader):
                mask = masks[
                    batch_id * batch_size : batch_id * batch_size + len(images)
                ].to(device)
                images = images.to(device)
                original_reps = basePMVAE.encoder(images)
                images = mask * images
                pert_reps = basePMVAE.encoder(images)
                rep_shift = torch.mean(
                    torch.sum((original_reps - pert_reps) ** 2, dim=-1)
                ).item()
                results_data.append([method_name, pert_percentage, rep_shift])

    print("Saving the plot")
    results_df = pd.DataFrame(
        results_data, columns=["Method", "% Perturbed Features", "Representation Shift"]
    )
    sns.set(font_scale=1.3)
    sns.set_style("white")
    sns.set_palette("colorblind")
    sns.lineplot(
        data=results_df,
        x="% Perturbed Features",
        y="Representation Shift",
        hue="Method",
    )
    plt.tight_layout()
    plt.savefig(save_dir / "pmvae_consistency_features.png")
    plt.close()


def consistency_examples(
    filename: str,
    random_seed: int = 1,
    batch_size: int = 256,
    dim_latent: int = 4,
    n_epochs: int = 2,
    subtrain_size: int = 1000,
) -> None:
    # Initialize seed and device
    # torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    (
        tr_ds,
        val_ds,
        test_ds,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        data_dim,
        membership_mask,
    ) = load_data(filename, device)

    checkpoint_dir = Path.cwd() / "results/vae"
    checkpoint_path = f"{checkpoint_dir}/{filename}_baseModel.pkl"

    save_dir = Path.cwd() / "results/vae/consistency_examples"

    train_loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    # Initialize encoder, decoder and autoencoder wrapper
    basePMVAE = pmVAEModel(
        membership_mask.values,
        [12],
        dim_latent,
        beta=1e-05,
        terms=membership_mask.index,
        add_auxiliary_module=False,
        use_gpu=True,
    )
    basePMVAE.train(
        tr_ds,
        val_ds,
        checkpoint_path=checkpoint_path,
        max_epochs=n_epochs,
        pathway_dropout=True,
        batch_size=batch_size,
    )

    subtrain_size = 2500
    idx_subtrain = random.sample(range(0, subtrain_size), subtrain_size)

    subtest_size = 1000
    idx_subtest = random.sample(range(0, subtest_size), subtest_size)

    train_subset = Subset(tr_ds, idx_subtrain)
    test_subset = Subset(test_ds, idx_subtest)

    subtrain_loader = DataLoader(train_subset)
    subtest_loader = DataLoader(test_subset)
    # labels_subtrain = torch.cat([label for _, label in subtrain_loader])
    # labels_subtest = torch.cat([label for _, label in subtest_loader])
    labels_subtrain = torch.tensor(y_train[idx_subtrain], dtype=torch.float32)
    labels_subtest = torch.tensor(y_test[idx_subtest], dtype=torch.float32)

    # Create a training set sampler with replacement for computing influence functions
    recursion_depth = 100
    train_sampler = RandomSampler(
        tr_ds, replacement=True, num_samples=recursion_depth * batch_size
    )
    train_loader_replacement = DataLoader(tr_ds, batch_size, sampler=train_sampler)

    # Fitting explainers, computing the metric and saving everything
    mse_loss = torch.nn.MSELoss()
    explainer_list = [
        # InfluenceFunctions(basePMVAE.encoder, mse_loss, save_dir / "if_grads"),
        # TracIn(basePMVAE, mse_loss, save_dir / "tracin_grads"),
        SimplEx(basePMVAE, mse_loss),
        NearestNeighbours(basePMVAE.encoder, mse_loss),
    ]
    frac_list = [0.05, 0.1, 0.2, 0.5, 0.7, 1.0]
    n_top_list = [int(frac * len(idx_subtrain)) for frac in frac_list]
    results_list = []
    for explainer in explainer_list:
        print(f"Now fitting {explainer} explainer")
        attribution = explainer.attribute_loader(
            device,
            subtrain_loader,
            subtest_loader,
            train_loader_replacement=train_loader_replacement,
            recursion_depth=recursion_depth,
        )
        # autoencoder.load_state_dict(
        #     torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
        # )
        sim_most, sim_least = similarity_rates(
            attribution, labels_subtrain, labels_subtest, n_top_list
        )
        results_list += [
            [str(explainer), "Most Important", 100 * frac, sim]
            for frac, sim in zip(frac_list, sim_most)
        ]
        results_list += [
            [str(explainer), "Least Important", 100 * frac, sim]
            for frac, sim in zip(frac_list, sim_least)
        ]
    results_df = pd.DataFrame(
        results_list,
        columns=[
            "Explainer",
            "Type of Examples",
            "% Examples Selected",
            "Similarity Rate",
        ],
    )
    print(f"Saving results in {save_dir}")
    results_df.to_csv(save_dir / "metrics.csv")
    sns.set(font_scale=1.3)
    sns.set_style("white")
    sns.set_palette("colorblind")

    sns.lineplot(
        data=results_df,
        x="% Examples Selected",
        y="Similarity Rate",
        hue="Explainer",
        style="Type of Examples",
        palette="colorblind",
    )
    plt.tight_layout()
    plt.savefig(save_dir / "similarity_rates.png")
    plt.savefig(save_dir / "pmvae_consistency_features.png")
    plt.close()


if __name__ == "__main__":
    # available_datasets = [
    #     "kang_PBMC",
    #     "norman_k562_crispr",
    #     "datlinger",
    #     "mcfarland",
    #     "haber_intestinal_epithelial",
    # ]
    filename = "haber_intestinal_epithelial"
    pretext_task_sensitivity(filename, n_epochs=1)
    consistency_feature_importance(filename, n_epochs=25)
    consistency_examples(filename, n_epochs=25)
