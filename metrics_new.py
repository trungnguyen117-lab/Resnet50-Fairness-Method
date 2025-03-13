import torch
from numpy import mean
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    precision_score, 
    f1_score,
    recall_score,
    classification_report
)
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, demographic_parity_difference, equalized_odds_difference

def get_metric_index():
    metric_index = ["acc", "bacc", "spd", "deo", "eod", "aaod", "aed", "precision", "recall", "f1", "time per epoch"]
    return metric_index

def _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight) -> MetricFrame:
    fns = {"tpr": true_positive_rate, "fpr": false_positive_rate}
    sw_dict = {"sample_weight": sample_weight}
    sp = {"tpr": sw_dict, "fpr": sw_dict}
    eo = MetricFrame(
        metrics=fns,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params=sp,
    )
    return eo

def equal_opportunity_difference(y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None) -> float:
    eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)
    return eo.difference(method=method)["tpr"]

def average_odds_difference(y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None) -> float:
    eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)
    tprd = eo.difference(method=method)["tpr"]
    fprd = eo.difference(method=method)["fpr"]
    return 0.5 * (tprd + fprd)

def accuracy_equality_difference(y_true, y_pred, sensitive_features):
    misclassification_rate_p = sum(y_true[sensitive_features == 1] != y_pred[sensitive_features == 1]) / sum(sensitive_features == 1)
    misclassification_rate_n = sum(y_true[sensitive_features == 0] != y_pred[sensitive_features == 0]) / sum(sensitive_features == 0)
    return abs(misclassification_rate_p - misclassification_rate_n)

def multiclass_ovr_fairness(y_true, y_pred, sensitive_features):
    num_classes = len(torch.unique(y_true))
    fairness_metrics = {}
    fairness_index = ["spd", "deo", "eod", "aaod", "aed"]
    for f_index in fairness_index:
        fairness_metrics[f_index] = []

    for cls in range(num_classes):
        y_true_cls = (y_true == cls).to(torch.float)
        y_pred_cls = (y_pred == cls).to(torch.float)

        spd_cls = demographic_parity_difference(y_true_cls, y_pred_cls, sensitive_features=sensitive_features)
        fairness_metrics["spd"].append(spd_cls)

        deo_cls = equalized_odds_difference(y_true_cls, y_pred_cls, sensitive_features=sensitive_features)
        fairness_metrics["deo"].append(deo_cls)

        eod_cls = equal_opportunity_difference(y_true_cls, y_pred_cls, sensitive_features=sensitive_features)
        fairness_metrics["eod"].append(eod_cls)

        aaod_cls = average_odds_difference(y_true_cls, y_pred_cls, sensitive_features=sensitive_features)
        fairness_metrics["aaod"].append(aaod_cls)

        aed_cls = accuracy_equality_difference(y_true_cls, y_pred_cls, sensitive_features=sensitive_features)
        fairness_metrics["aed"].append(aed_cls)

    spd = mean(fairness_metrics["spd"])
    deo = mean(fairness_metrics["deo"])
    eod = mean(fairness_metrics["eod"])
    aaod = mean(fairness_metrics["aaod"])
    aed = mean(fairness_metrics["aed"])

    return spd, deo, eod, aaod, aed

def get_all_metrics(y_true, y_pred, sensitive_features):
    ret = {}

    ret["acc"] = accuracy_score(y_true, y_pred)
    ret["bacc"] = balanced_accuracy_score(y_true, y_pred)

    ret["spd"], ret["deo"], ret["eod"], ret["aaod"], ret["aed"] = multiclass_ovr_fairness(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
    )

    # Calculate precision, recall and F1 score
    ret["precision"] = precision_score(y_true, y_pred, average='weighted')
    ret["recall"] = recall_score(y_true, y_pred, average='weighted')
    ret["f1"] = f1_score(y_true, y_pred, average='weighted')
    
    # Get classification report
    ret["classification_report"] = classification_report(y_true, y_pred)

    return ret

def print_all_metrics(ret):
    print("\n========================== Performance and Fairness Metrics ==========================\n")
    print("Accuracy: %.3f\n" % ret["acc"])
    print("Balanced Accuracy: %.3f\n" % ret["bacc"])
    print("Statistical Parity Difference (SPD): %.3f\n" % ret["spd"])
    print("Equalized Odds Difference (DEO): %.3f\n" % ret["deo"])
    print("Equal Opportunity Difference (EOD): %.3f\n" % ret["eod"])
    print("Average Absolute Odds Difference (AAOD): %.3f\n" % ret["aaod"])
    print("Accuracy Equality Difference (AED): %.3f\n" % ret["aed"])
    print("Precision: %.3f\n" % ret["precision"])
    print("Recall: %.3f\n" % ret["recall"])
    print("F1 Score: %.3f\n" % ret["f1"])
    print("\n========================== Classification Report ==========================\n")
    print(ret["classification_report"])

if __name__ == "__main__":
    y_true = torch.tensor([0, 1, 1, 1, 2, 0, 2, 0, 1, 0, 1, 2])  # Example for 3-class classification
    y_pred = torch.tensor([2, 1, 1, 0, 2, 1, 1, 0, 0, 2, 0, 1])
    sf_data = torch.tensor([1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1])

    ret = get_all_metrics(y_true, y_pred, sensitive_features=sf_data)
    print_all_metrics(ret)