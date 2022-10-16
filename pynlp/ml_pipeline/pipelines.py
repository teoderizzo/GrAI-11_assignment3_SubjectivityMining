from sklearn.pipeline import Pipeline, FeatureUnion
from ml_pipeline import preprocessing, representation
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm



def pipeline(preprocessor, representation, classifier):
    return Pipeline([('prep', preprocessor),
                     ('frm', representation),
                     ('clf', classifier)])


def combined_pipeline(prep1, repr1, prep2, repr2, classifier):
    combined_features = FeatureUnion([
        ('token_features', Pipeline([('prep1', prep1), ('repr1', repr1)])),
        ('polarity_features', Pipeline([('prep2', prep2), ('repr2', repr2)]))])
    return Pipeline([('features', combined_features),
                     ('clf', classifier)])



# ------------- parametrization ---------------------------

def svm_clf_grid_parameters():
    """Example parameters for svm.LinearSVC grid search

    The preprocessor and formatter can also be parametrized through the prefixes 'prep' and 'frm', respectively."""
    return {'clf__class_weight': (None, 'balanced'),
            'clf__dual': (True, False),
            'clf__C': (0.1, 1, 10)}



#------------------------ pipeline ------------------------

def svm_libsvc_counts_12():
    return pipeline(preprocessing.std_prep(), representation.count_vectorizer({'min_df': 1, 'max_df':0.5, 'ngram_range': (1,2)}),
                    svm.LinearSVC(max_iter=10000, dual=False, C=0.5))

