# Elisabeth Holm implementation of ML algorithms for text classification of emails
import os
import random
import ast
import numpy as np
import math

trial = 1

allpercents_standard_nb = []  # naive bayes (method 1 -- standard libraries)
allpercents_nb = []  # naive bayes (method 2 -- scikit)
allpercents_svm = []  # support vector machines
allpercents_gs = []  # grid search
allpercents_lr = []  # logistic regression

percent_train = .90  # 90% of data is used for training

while trial <= 1:  # make everything run 1 times

    ticket_bodies = []
    ticket_nums = []
    ticket_dicts = []
    categories = []
    target = []
    filepath = "//home/elisabeth/rehs/rehs_text_class"  # change this if you are using a different computer

    # creates categories[]
    def get_categories():
        for category in os.listdir(filepath + "/training_emails"):  # goes into categories folder, and adds each category to
            # commented out portions below only use certain categories based on amount of data or by name
            #if len(os.listdir(filepath + "/training_emails/" + category)) > 43 and len(os.listdir(filepath + "/training_emails/" + category)) < 67:
            #if (category == "storage") or (category == "no_dir") or (category == "user_not_found") or (category == "vasp") or (category == "unavailable_node") or (category == "matlab") or (category == "lammps") or (category == "extend_wallclock") or (category == "disk_quota") or (category == "disallowed"):
            categories.append(category)  # a list of the categories in order of appearance in training_emails


    # creates ticket_bodies[], ticket_nums[], and ticket_dicts[]
    def get_ticket_info(current_num, cat):
        for UID in os.listdir(filepath + "/training_emails/" + cat + "/ticket#" + current_num):
            try:
                dict = open(
                    filepath + "/training_emails/" + cat + "/ticket#" + current_num + "/" + UID + "/" + "dictionary.txt",
                    "r")
                ticket_dicts.append(ast.literal_eval(dict.read()))  # adds the dictionary as a dict into an list

                body_stripped = open(
                    filepath + "/training_emails/" + cat + "/ticket#" + current_num + "/" + UID + "/" + "body_stripped.txt",
                    "r")
                body_text = body_stripped.read()
                ticket_bodies.append(body_text)  # adds the body_stripped text into an list
                ticket_nums.append(current_num)  # adds that ticket number into an list in the same
                # index as its corresponding body_stripped.txt
            except:
                pass  # if there is no dictionary or body_stripped file for that ticket for some reason, skip it
        return ticket_bodies, ticket_nums

    get_categories()

    # creates an list for the machine to check against (based on index in the list, can see the correct category)
    def get_targets():
        for cat in categories:
            # go into that category's folder in training_emails
            for tix in os.listdir(filepath + "/training_emails" + "/" + cat):
                t = tix.strip("ticket#")  # current ticket number
                get_ticket_info(t, cat)
                index = ticket_nums.index(t)  # index of the ticket number in ticket_nums
                target.insert(index, cat)

    get_targets()

    # the next 15 lines scramble the data and split it into train and test data
    data = list(zip(ticket_bodies, ticket_nums, target, ticket_dicts))  # combines all corresponding data into a single list
    random.shuffle(data)  # all of this makes sure all data has corresponding indexes for the different info
    test_data = data[round(len(data) * percent_train):]  # test data uses last 10% of shuffled tickets
    train_data = data[:round(len(data)*percent_train)]  # train data uses first 90% of shuffled tickets

    tt_b, tt_n, tt, tt_d = zip(*train_data)  # unzips (separates) the data and splits them into different lists for later use
    train_ticket_bodies = list(tt_b)
    train_ticket_nums = list(tt_n)
    train_target = list(tt)
    train_ticket_dicts = list(tt_d)

    t_b, t_n, t, t_d = zip(*test_data)
    test_ticket_bodies = list(t_b)
    test_ticket_nums = list(t_n)
    test_target = list(t)
    test_ticket_dicts = list(t_d)

    # from here on,the code is actually implementing the text classification machine learning algorithms
    ####################################################################################################################


    #METHOD 1: MULTINOMIAL NAIVE BAYES STANDARD LIBRARIES ONLY

    # creates the document term frequency matrix
    def create_dtfm(dicts):
        # dicts is each ticket's dictionary in a list
        dtfm = []  # document term frequency matrix
        totalwords = []
        dtfm.append([])
        for current_dict in dicts:
            total = 0
            for word, freq in current_dict.items():
                total += freq
                try:  # if the the word is already in the dtfm matrix
                    word_index = dtfm[0].index(word)  # if this returns an error you know its not in the known words yet
                except:  # if the word isn't in the known words yet, add it
                    dtfm[0].append(word)
            totalwords.append(total)  # total words in the document (for use later)

        empty_row = [0] * (len(dtfm[0])+1)
        for a in dicts:
            dtfm.append(empty_row[:])

        current_row = 0
        while current_row < len(dicts):
            current_row += 1
            current_dict = dicts[current_row-1]
            for word, freq in current_dict.items():
                # add the frequency
                word_index = dtfm[0].index(word)
                dtfm[current_row][word_index] = (freq/totalwords[current_row-1])
                # by doing freq/totalwords, it weighs the word relative to the length of the document to account for
                # longer vs shorter documents
            dtfm[current_row][len(dtfm[current_row])-1] = sum(dtfm[current_row])
            # adds up word total at the end of each row
        return dtfm



    # creates the dtfm without using the dictionaries provided
    # currently this function is not being used, as we have a dictionary.txt file for each ticket
    # but it may be of use later when only given stripped text
    def create_dtfm_without_dicts(bodies):
        # bodies = stripped ticket bodies in a list
        dtfm = []
        dtfm.append([])  # adds an empty row at the beginning for the vocabulary
        for b in bodies:
            words = b.split()
            for word in words:  # look at each word individually
                try:
                    dtfm[0].index(word)  # if the word isn't already in the vocabulary, this will give an error
                except:
                    dtfm[0].append(word)  # when it finds a word that's not in the vocabulary, add the word to the vocab

        empty_row = [0] * (len(dtfm[0]) + 1)
        for a in bodies:
            dtfm.append(empty_row[:])  # append one empty row (the length of the vocabulary + 1 for totals) per document

        current_row = 0
        for current_body in bodies:
            current_row += 1
            words = current_body.split()
            for word in words:
                # add the frequency
                word_index = dtfm[0].index(word)
                dtfm[current_row][word_index] += 1
                dtfm[current_row][len(dtfm[current_row])-1] += 1
                # adds up word total at the end of each row
            i = 0
            for num in dtfm[current_row]:
                row_total = dtfm[current_row][len(dtfm[current_row])-1]
                if num == row_total:
                    dtfm[current_row][i] = sum(dtfm[current_row]) - row_total
                elif num == 0:
                    pass
                else:
                    dtfm[current_row][i] = num/row_total  # make the frequency relative to the length of the document
                    # to account for long vs short documents (long documents would just have more words and occurances)
                i += 1
        return dtfm



    # creates a categorized version of dtfm for use in recognizing common patterns in individual categories
    def categorize_dtfm(dtfm):
        categorized_dtfm = []  # clears any data from before
        # mode is train or test
        for a in categories:
            categorized_dtfm.append([])
        for doc in dtfm:
            if dtfm.index(doc) == 0:
                pass
            else:
                doc_index = dtfm.index(doc)
                cat = train_target[doc_index-1]  # get the corresponding correct category by corresponding indexes
                cat_index = categories.index(cat)  # find what index the category is (same index as in categorized_dtfm)
                categorized_dtfm[cat_index].append(doc)  # add the row from dtfm to its corresponding category in categorized_dtfm

        empty_row = [0] * (len(dtfm[1]))
        for cat in categorized_dtfm:
            cat.append(empty_row[:])
            totals = cat[len(cat)-1]  # last row of each category (the totals for each word)
            for doc in cat:
                if doc == totals:
                    continue
                word_index = 0
                for word_freq in doc:
                    totals[word_index] += word_freq  # totals the amount of times the word appears in that category
                    word_index += 1

        return categorized_dtfm



    # makes the predictions for classification based on the training and testing data
    def predict(matrix, train, vocab):
        # matrix = uncategorized dtfm for test data
        # train = categorized training data
        # vocab is training data's vocabulary
        predictions = []
        predictions.clear()

        priors = []
        for c in train:
            priors.append((len(c)-1)/len(train_ticket_dicts))  # num of docs in each category / total number of docs
        probabilities = []

        for doc in matrix:
            probabilities.clear()
            if doc == matrix[0]:  # skip the first "doc", which is the vocab list
                continue
            y = []
            cat_index = 0
            for cat in categories:
                totals = train[cat_index][len(train[cat_index])-1]  # last row of whatever category you're on
                y.clear()
                test_word_index = 0
                for word_freq in doc:
                    # skip word_freq if its the last one in the doc (the total) or if it doesn't appear in the document
                    if test_word_index == (len(doc)-1) or word_freq == 0:
                        test_word_index += 1
                    else:
                        word = matrix[0][test_word_index]  # the actual word (not just the number of times it appears
                        try:
                            vocab_index = vocab.index(word)  # index of word in training vocabulary
                            # x is the probability of the word given the class (w|c)
                            x = (totals[vocab_index] + .001) / (totals[len(totals) - 1] + len(vocab))
                            #  (count of that word appearing in that class + smoothing variable) /
                            #  (count of all words in that class + length of vocabulary
                            y.append(-math.log(x))  # append the probability of that word given that category
                            test_word_index += 1
                        except:  # if the word doesn't appear in the training data's vocabulary, skip it
                            continue

                prob = sum(y) - math.log(priors[cat_index])
                probabilities.append(prob)  # appends the probability of that doc being in that category

                cat_index += 1


            cat_index = 0
            min = probabilities[0]
            for p in probabilities:
                if p < min:
                    min = p  # find the minimum probability
                    cat_index = probabilities.index(p)  # save the index of it and use that once the max is found (line below)
            predictions.append(categories[cat_index])  # append the most probable category

        return predictions



    # training
    train_dtfm = create_dtfm(train_ticket_dicts)
    training_categorized_dtfm = categorize_dtfm(train_dtfm)

    # testing
    test_dtfm = create_dtfm(test_ticket_dicts)
    predicted_nb = predict(test_dtfm, training_categorized_dtfm, train_dtfm[0])

    # after the program has made its predictions, calculate average correct
    correct = 0
    index = 0
    for p in predicted_nb:
        if p == test_target[index]:
            correct += 1
        index += 1

    accuracy_s_nb = correct/len(predicted_nb)
    allpercents_standard_nb.append(accuracy_s_nb*100)




    #METHOD 2: SCIKIT MULTINOMIAL NAIVE BAYES
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    # above imports are used in methods 2-4 as well
    from sklearn.naive_bayes import MultinomialNB

    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=1.0, fit_prior=False, class_prior=None))])
    text_clf = text_clf.fit(train_ticket_bodies, train_target)

    predicted_scikit_nb = text_clf.predict(test_ticket_bodies)
    accuracynb = np.mean(predicted_scikit_nb == test_target)

    allpercents_nb.append(accuracynb*100)



    #METHOD 3: SCIKIT NAIVE BAYES + GRID SEARCH
    from sklearn.model_selection import GridSearchCV
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3)}
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_ticket_bodies, train_target)

    accuracygs = gs_clf.best_score_
    allpercents_gs.append(accuracygs * 100)



    #METHOD 4: SCIKIT SUPPORT VECTOR MACHINES
    from sklearn.linear_model import SGDClassifier
    text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha = 1e-3, n_iter = 5, random_state = 42))])
    _ = text_clf_svm.fit(train_ticket_bodies, train_target)

    predicted_svm = text_clf_svm.predict(test_ticket_bodies)
    accuracysvm = np.mean(predicted_svm == test_target)

    allpercents_svm.append(accuracysvm*100)


    '''
    #METHOD 5: SCIKIT LOGISTIC REGRESSION -- Does not currently work but something to look into
    # import the class
    from sklearn.linear_model import LogisticRegression
    # instantiate the model (using the default parameters)
    logisticRegr = LogisticRegression()
    # fit the model with data
    logisticRegr.fit(train_dtfm[1:], train_target)
    predicted_lr = logisticRegr.predict(create_dtfm(test_ticket_dicts)[1:])

    accuracylr = np.mean(predicted_lr == test_target)
    allpercents_lr.append(accuracylr * 100)
    '''


    # for use in creating confusion matrices using sam_confusion_matrix.py
    def get_results(method):
        if method == "NB":
            return predicted_nb, test_target
        elif method == "SCIKIT NB":
            return predicted_scikit_nb, test_target
        elif method == "SCIKIT SVM":
            return predicted_svm, test_target
        else:
            raise Exception("Invalid machine learning method, please enter one of the following: NB, SCIKIT NB, SCIKIT NB GS, SCIKIT SVM")


    trial = trial + 1


# print out the results for each method
print()
print()
print()
print("All accuracy results for standard library NB:" + str(allpercents_standard_nb))
avg_s_nb = sum(allpercents_standard_nb) / len(allpercents_standard_nb)
print("Program ran: " + str(trial-1) + " times at " + str(round(percent_train*100)) + "% training data")
print("Average: " + str(avg_s_nb) + "% correct")
print()

print("All accuracy results for scikit NB:" + str(allpercents_nb))
avg_nb = sum(allpercents_nb) / len(allpercents_nb)
print("Program ran: " + str(trial-1) + " times at " + str(round(percent_train*100)) + "% training data")
print("Average: " + str(avg_nb) + "% correct")
print()

print("All accuracy results for scikit NB + GS:" + str(allpercents_gs))
avg_gs = sum(allpercents_gs) / len(allpercents_gs)
print("Program ran: " + str(trial-1) + " times at " + str(round(percent_train*100)) + "% training data")
print("Average: " + str(avg_gs) + "% correct")
print()

print("All accuracy results for scikit SVM:" + str(allpercents_svm))
avg_svm = sum(allpercents_svm) / len(allpercents_svm)
print("Program ran: " + str(trial-1) + " times at " + str(round(percent_train*100)) + "% training data")
print("Average: " + str(avg_svm) + "% correct")
print()

'''
#if logistic regression gets working, uncomment this portion to display accuracy results
print("All accuracy results for scikit LR:" + str(allpercents_lr))
avg_lr = sum(allpercents_lr) / len(allpercents_lr)
print("Program ran: " + str(trial-1) + " times at " + str(round(percent_train*100)) + "% training data")
print("Average: " + str(avg_lr) + "% correct")
'''