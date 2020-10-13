# Supervised-Machine-Learning-Text-Classification
Project: REHS Text Classification via Supervised Machine Learning for an Issue Tracking System
Mentor: Dr.Martin Kandes
Organization: San Diego Supercomputer Center

User Guide for ElisabethImplementation.py
Code and User Guide by Elisabeth Holm

Information-Getting Functions:
•	get_categories()- creates a list of all the categories (categories[]); in order to play with the amount of categories or which specific ones to go through, you can change the categories list and the rest of the code will continue to run smoothly

•	get_ticket_info(current_num, cat)- creates lists ticket_bodies[], ticket_nums[], and ticket_dicts[], three lists with corresponding indexes for each ticket in each category with the body, ticket number, and dictionary for that ticket; parameters are used because this function is called during an iteration through all the tickets in all the categories in get_targets()

•	get_targets()- creates list target[] which has, in corresponding index, the correct category for each ticket (ex: [disallowed, disallowed, trial_account, vi_slow])

•	lines 70-86, unofficially "scramble()" - zips the different lists with individual ticket information in corresponding indexes (ticket_bodies[], ticket_nums[], target[], and ticket_dicts[]) using the built in zip( ) function so the corresponding index information stays together, then shuffles the data, splits the large zipped list into training and testing data (first x% is for training, last (100 - x)% is for testing), and reorganizes the ticket information into lists train_ticket_bodies[], train_ticket_nums[], train_target[], train_ticket_dicts[], test_ticket_bodies[], test_ticket_nums[], test_target[], and test_ticket_dicts[], which are organized the same as the inital bodies, nums, and target lists, just with a difference between the train and test ticket groups

•	get_results(method) - returns the prediction[] and target[] lists for whichever classifying method is passed into the header ("NB", "SCIKIT NB", or "SCIKIT SVM"). For use when getting the results from ElisabethImplementation.py to another class, such as sam_confusion_matrix.py, which creates and saves a confusion matrix based on the given target and prediction lists

Naive Bayes-Specific Functions:
•	create_dtfm(dicts) - creates the document term frequency matrix (shortened to "dtfm") given a list of dictionaries in the parameters (such as train_ticket_dicts[] or test_ticket_dicts[]); returns a list of lists with the first element being a list of the unique vocabulary words and all elements after that being a list of numbers for frequency of words for every ticket with a total of amount of words at the end of the list (ex: [[word, bleh, wow], [1, 0, 0, 1], [0, 2, 1, 3], [1, 1, 1, 3]])*

•	create_dtfm_without_dicts(bodies) - creates and returns the same document term frequency matrix as the previous function but when only given a list of ticket bodies, not a list of dictionaries; currently this function is not being used because all the tickets have dictionaries, but it may be useful in the future when we may be simply given text and not a premade dictionary for each ticket

•	categorize_dtfm(dtfm) - creates and returns a categorized document term frequency matix using the previously made document term frequency matrix that is passed in with a total for each word at the end of each category and no vocabulary list. Ex with 3 categories:[[[1, 0, 1, 2], [3, 0, 0, 3], [2, 1, 1, 4], [6, 1, 2, 9]],  [[2, 1, 2, 5], [1, 1, 1, 3], [3, 2, 3, 8]],  [[1, 1, 2, 4][2, 1, 3, 6][1, 1, 2, 4][2, 1, 3, 6][6, 4, 10, 20]]]*. This is used for the dtfm for the training data only, since that is the only one we "know" the correct categories for. This categorized dtfm is used later in the naive bayes equation.

•	predict(matrix, train, vocab) - uses the Naive Bayes equation to predict the probability of each ticket being in each category, finds the most probably category for each ticket, then returns list predictions[] with each ticket's predicted category (ex: [disallowed, vi_slow, disallowed, install_update, trial_account])

*The lists in the examples are using whole numbers for simplicity, but in reality each number is divided by the total number of words for that ticket/category (in the category totals) as to not give tickets with more words or categories with more documents higher weight/importance


Alternate/Advanced Implemented Methods (All created using scikit at the bottom of the code (lines 285-386)):
•	Naive Bayes (using scikit instead)
•	Naive Bayes + Grid Search
•	Support Vector Machines (SVM)
•	Logistic Regression (Does not currently work but could be something to look into further, currently commented out to avoid code stopping due to errors)



*This code was created for a 2019 summer internship at the Supercomputer center.
