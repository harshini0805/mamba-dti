import torch

from utils.metrics import compute_metrics


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def run_epoch(
    model,
    dataloader,
    criterion,
    optimizer=None,
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0

    labels_all = []

    probabilities_all = []

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for protein, drug, labels in dataloader:

            protein = protein.to(DEVICE)

            drug = drug.to(DEVICE)

            labels = labels.to(DEVICE)

            logits = model(
                protein,
                drug,
            )

            loss = criterion(
                logits,
                labels,
            )

            if training:

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()

            running_loss += loss.item() * labels.size(0)

            probabilities = torch.sigmoid(logits)

            probabilities_all.extend(
                probabilities.detach().cpu().numpy()
            )

            labels_all.extend(
                labels.cpu().numpy()
            )

    epoch_loss = running_loss / len(dataloader.dataset)

    metrics = compute_metrics(
        labels_all,
        probabilities_all,
    )

    return epoch_loss, metrics


def predict(model, dataloader):
    """
    Run inference over a dataloader and return raw (labels, probabilities)
    with no threshold applied and no loss computed.

    Used after training completes to (a) get validation probabilities for
    threshold search, and (b) get test probabilities to evaluate at a
    tuned threshold. Kept separate from run_epoch so the per-epoch
    training/validation hot loop is untouched.
    """

    model.eval()

    labels_all = []
    probabilities_all = []

    with torch.no_grad():

        for protein, drug, labels in dataloader:

            protein = protein.to(DEVICE)

            drug = drug.to(DEVICE)

            logits = model(
                protein,
                drug,
            )

            probabilities = torch.sigmoid(logits)

            probabilities_all.extend(
                probabilities.detach().cpu().numpy()
            )

            labels_all.extend(
                labels.numpy()
            )

    return labels_all, probabilities_all
