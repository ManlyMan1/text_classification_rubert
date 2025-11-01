import datetime
import random

import pandas as pd
import sklearn.utils
import stopwatch
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, \
    classification_report
from transformers import BertTokenizer

from bert_classifier import BertClassifier

st = stopwatch.Stopwatch()
torch.cuda.empty_cache()


log_file_name = ("logs\\log_BERT_" + str(datetime.datetime.now()) + ".txt").replace(":", "-")
with open(log_file_name, mode="a", encoding="utf8") as file:
    file.write("\n\n\n\n\t\tNew process --------------------------------------------------\n\n\n\n")


def fprint(data=""):
    with open(log_file_name, mode="a", encoding="utf8") as file:
        if str(data) != "":
            file.write(str(datetime.datetime.now()) + ":\n")
            file.write(str(data).replace('.', ',') + "\n\n")
        else:
            file.write("\n")
    if str(data) != "":
        print(str(datetime.datetime.now()) + ":\n" + str(data))
    print()


plt.figure(figsize=(7, 7))


def draw_roc(TestY, lr_probs, model_name, id):
    fpr, tpr, treshold = roc_curve(TestY, lr_probs)
    roc_auc = auc(fpr, tpr)
    # строим график
    # plt.plot(fpr, tpr, color=colors_list[random.randint(0, len(colors_list)-1)],
    #          label='ROC кривая (area = %0.2f)' % roc_auc)
    plt.plot(fpr, tpr, color=(random.randint(0, 100) / 255, random.randint(0, 100) / 255, random.randint(0, 100) / 255),
             label='ROC кривая (area = %0.2f), column ' % roc_auc + str(id), linewidth=2)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str(model_name) + "\n" + str(datetime.datetime.now()))
    plt.legend(loc="lower right")
    plt.savefig("images\\" + str(model_name) + " " + str(datetime.datetime.now()).replace(":", "-") + ".png")
    # plt.clf()


model_name = 'cointegrated/rubert-tiny'
tokenizer = BertTokenizer.from_pretrained(model_name)
max_tokens = tokenizer.model_max_length


def main():
    # train_data = pd.read_csv('/content/train.csv')
    # valid_data = pd.read_csv('/content/valid.csv')
    # test_data = pd.read_csv('/content/test.csv')

    max_tokens = 512
    # logging.basicConfig(level=logging.INFO)
    # transformers_logger = logging.getLogger("transformers")
    # transformers_logger.setLevel(logging.WARNING)
    average_types = ['weighted_by_non_zeroes']
    for average_method in average_types:
        fprint(average_method)
        classifier = BertClassifier(
            model_path='cointegrated/rubert-tiny',
            tokenizer_path='cointegrated/rubert-tiny',
            n_classes=2,
            epochs=3,
            model_save_path='content/bert.pt'
        )


        predictors = {1: 4, 2: 5, 3: 2}

        plt.clf()
        #average_type = 'weighted_by_average'
        for p1 in predictors.keys():
            for p2 in range(1, predictors[p1] + 1):
                col = str(p1) + "." + str(p2) + '_'
                fprint(str(p1) + "." + str(p2) + '_')
                df_data = pd.read_excel('level_1_' + average_method + '4.xlsx', nrows=18939)

                fprint("Количество данных во входном файле")
                fprint(df_data.shape[0])
                coef = 1
                if coef < 1:
                    df_data = df_data.sample(frac=coef)
                    fprint("Количество данных после применения множителя (текущий множитель = " + str(coef) + ")")
                    fprint(df_data.shape[0])

                df_data["token_count"] = df_data["Text"].apply(lambda x: len(tokenizer.tokenize(x)))
                df_data = df_data.drop(df_data[df_data.token_count >= max_tokens - 5].index)
                fprint(
                    "Количество данных после фильтрации по количеству токенов (ограничение = " + str(max_tokens) + ")")
                fprint(str(df_data.shape[0]) + "\n")


                zeroes = df_data[df_data[col] == 0]
                ones = df_data[df_data[col] == 1]
                if zeroes.shape[0] > ones.shape[0]:
                    zeroes = sklearn.utils.resample(zeroes, replace=True,
                                                    n_samples=len(ones),
                                                    random_state=42)
                else:
                    ones = sklearn.utils.resample(ones, replace=True,
                                                  n_samples=len(zeroes),
                                                  random_state=42)
                df_data = pd.concat([zeroes, ones], axis=0)
                df_data = df_data.sample(frac = 1)
                fprint(df_data[col].value_counts())
                train_size = int(df_data.shape[0] * 0.7)
                val_size = df_data.shape[0] - train_size
                df_data_train, df_data_val = torch.utils.data.random_split(df_data, [train_size, val_size])
                df_data_train = df_data.iloc[df_data_train.indices]
                df_data_val = df_data.iloc[df_data_val.indices]
                train_data = pd.DataFrame()
                eval_data = pd.DataFrame()

                train_data['text'] = df_data_train['Text']
                train_data['label'] = df_data_train[str(p1) + "." + str(p2) + '_']

                eval_data['text'] = df_data_val['Text']
                eval_data['label'] = df_data_val[str(p1) + "." + str(p2) + '_']

                # cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                auc_scores = []
                st.reset()
                st.start()
                max_auc_score = 0
                id = 1

                X_train, X_val = train_data["text"], eval_data["text"]
                y_train, y_val = train_data['label'], eval_data['label']

                classifier.preparation(
                    X_train=list(X_train),
                    y_train=list(y_train),
                    X_valid=list(X_val),
                    y_valid=list(y_val)
                )

                classifier.train()
                # Train the model on the training data
                # lr_model.fit(X_train, y_train)
                # y_pred = lr_model.predict(X_val)
                y_pred = [classifier.predict(t) for t in X_val]
                # Print the classification report
                fprint(classification_report(y_val, y_pred))
                # Predict probabilities for the positive class on the validation data
                probabilities = classifier.predict_proba(X_val)
                probabilities_ = list()
                for item in probabilities:
                    probabilities_.append(item[1])
                accuracy = accuracy_score(y_val, y_pred)
                fprint("accuracy: " + str(accuracy))
                draw_roc(y_val, probabilities_, "RuBERT", col)

                id += 1
                # Calculate ROC AUC score for the validation set
                auc_score = roc_auc_score(y_val, probabilities_)
                if auc_score > max_auc_score:
                    max_auc_score = auc_score
                    torch.save(classifier.model, "content/" + str(col) + "bert_best.pt")
                    # lr_model = joblib.load(joblib_file)
                auc_scores.append(auc_score)
                fprint(f'ROC AUC for fold ' + str(id) + ': ' + str(round(auc_score, 4)))
                st.stop()
                fprint("BERT, elapsed time = " + str(st.duration))
                # Print the scores for each fold
                for i, score in enumerate(auc_scores, 1):
                    fprint(f'ROC AUC for fold ' + str(i) + ': ' + str(round(score, 4)))

                fprint('Average ROC AUC:' + str(round(sum(auc_scores) / len(auc_scores), 4)))
                fprint('Standard deviation:' + str(
                    round(
                        (sum([(x - sum(auc_scores) / len(auc_scores)) ** 2 for x in auc_scores]) / len(
                            auc_scores)) ** 0.5,
                        4)))
                fprint()
                # load the best model
                classifier.model = torch.load("content/" + str(col) + "bert_best.pt", weights_only=False)

        for p1 in predictors.keys():

            col = str(p1) + '_'
            fprint(str(p1) + '_')

            df_data = pd.read_excel('level_1_' + average_method + '.xlsx', nrows=18939)

            fprint("Количество данных во входном файле")
            fprint(df_data.shape[0])
            coef = 1
            if coef < 1:
                df_data = df_data.sample(frac=coef)
                fprint("Количество данных после применения множителя (текущий множитель = " + str(coef) + ")")
                fprint(df_data.shape[0])

            df_data["token_count"] = df_data["Text"].apply(lambda x: len(tokenizer.tokenize(x)))
            df_data = df_data.drop(df_data[df_data.token_count >= max_tokens - 5].index)
            fprint(
                "Количество данных после фильтрации по количеству токенов (ограничение = " + str(max_tokens) + ")")
            fprint(str(df_data.shape[0]) + "\n")

            zeroes = df_data[df_data[col] == 0]
            ones = df_data[df_data[col] == 1]
            if zeroes.shape[0] > ones.shape[0]:
                zeroes = sklearn.utils.resample(zeroes, replace=True,
                                                n_samples=len(ones),
                                                random_state=42)
            else:
                ones = sklearn.utils.resample(ones, replace=True,
                                              n_samples=len(zeroes),
                                              random_state=42)
            df_data = pd.concat([zeroes, ones], axis=0)
            df_data = df_data.sample(frac=1)
            fprint(df_data[col].value_counts())

            train_size = int(df_data.shape[0] * 0.7)
            val_size = df_data.shape[0] - train_size
            df_data_train, df_data_val = torch.utils.data.random_split(df_data, [train_size, val_size])
            df_data_train = df_data.iloc[df_data_train.indices]
            df_data_val = df_data.iloc[df_data_val.indices]
            train_data = pd.DataFrame()
            eval_data = pd.DataFrame()

            train_data['text'] = df_data_train['Text']
            train_data['label'] = df_data_train[str(p1) + '_']

            eval_data['text'] = df_data_val['Text']
            eval_data['label'] = df_data_val[str(p1) + '_']

            # cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            auc_scores = []
            st.reset()
            st.start()
            max_auc_score = 0
            id = 1

            X_train, X_val = train_data["text"], eval_data["text"]
            y_train, y_val = train_data['label'], eval_data['label']

            classifier.preparation(
                X_train=list(X_train),
                y_train=list(y_train),
                X_valid=list(X_val),
                y_valid=list(y_val)
            )

            classifier.train()
            # Train the model on the training data
            # lr_model.fit(X_train, y_train)
            # y_pred = lr_model.predict(X_val)
            y_pred = [classifier.predict(t) for t in X_val]
            # Print the classification report
            fprint(classification_report(y_val, y_pred))
            # Predict probabilities for the positive class on the validation data
            probabilities = classifier.predict_proba(X_val)
            probabilities_ = list()
            for item in probabilities:
                probabilities_.append(item[1])
            accuracy = accuracy_score(y_val, y_pred)
            fprint("accuracy: " + str(accuracy))
            draw_roc(y_val, probabilities_, "RuBERT", col)

            id += 1
            # Calculate ROC AUC score for the validation set
            auc_score = roc_auc_score(y_val, probabilities_)
            if auc_score > max_auc_score:
                max_auc_score = auc_score
                torch.save(classifier.model, "content/" + str(col) + "bert_best.pt")
                # lr_model = joblib.load(joblib_file)
            auc_scores.append(auc_score)
            fprint(f'ROC AUC for fold ' + str(id) + ': ' + str(round(auc_score, 4)))
            st.stop()
            fprint("BERT, elapsed time = " + str(st.duration))
            # Print the scores for each fold
            for i, score in enumerate(auc_scores, 1):
                fprint(f'ROC AUC for fold ' + str(i) + ': ' + str(round(score, 4)))

            fprint('Average ROC AUC:' + str(round(sum(auc_scores) / len(auc_scores), 4)))
            fprint('Standard deviation:' + str(
                round(
                    (sum([(x - sum(auc_scores) / len(auc_scores)) ** 2 for x in auc_scores]) / len(auc_scores)) ** 0.5,
                    4)))
            fprint()
            # load the best model
            classifier.model = torch.load("content/" + str(col) + "bert_best.pt", weights_only=False)

    #######################################################################################################  uncomment
    # n = int(0.8 * len(train_data))
    # fprint(train_data.head(10))
    # x_train = train_data["text"].values[:n]
    # y_train = train_data["label"].values[:n].astype(int)
    # x_test = train_data["text"].values[n:]
    # y_test = train_data["label"].values[n:].astype(int)

    # texts = pd.read_excel("data/psych_texts/texts_256.xlsx")
    #
    # done = 0
    # part = 0
    # # gen_clean = Parallel(n_jobs=cpus, verbose=60)(delayed(preprocess)(x) for x in texts[texts.columns[0]])
    # # fprint("finished preprocessing texts")
    #
    # # texts["clean"] = gen_clean
    # texts.rename(columns={texts.columns[0]: 'text'}, inplace=True)
    # texts.rename(columns={"text": "text", "type": "label"}, inplace=True)
    # texts["label"] = texts["label"].apply(rename_to_zeorones)


#######################################################################################################  uncomment
# classifier.load_preparation(
#     X_train=list(x_train),
#     y_train=list(y_train),
#     X_valid=list(x_test),
#     y_valid=list(y_test)
# )
#
# classifier.load()

#
# texts_ = list(eval['clean'])
# labels = list(eval['generated'])
# plt.clf()
# predictions = [classifier.predict(t) for t in texts_]
#
# precision, recall, f1score = precision_recall_fscore_support(labels, predictions, average='macro')[:3]
# probabilities = classifier.predict_proba(texts_)
# probabilities_ = list()
# for item in probabilities:
#     probabilities_.append(item[1])
# auc_score = roc_auc_score(eval["generated"], probabilities_)
# draw_roc(eval["generated"], probabilities_, "RuBERT", "Evaluation")
# accuracy = accuracy_score(labels, predictions)
# fprint(f'precision: {precision}, recall: {recall}, f1score: {f1score}, accuracy: {accuracy}, auc score: {auc_score}')


main()
