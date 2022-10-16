import logging
import sys

from tasks import vua_format as vf
from ml_pipeline import utils, preprocessing
from ml_pipeline import pipelines


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

#-------------------------- runs the model --------------------------------------

def run(task_name, data_dir, pipeline_name, print_predictions):
    logger.info('>> Running {} experiment'.format(task_name))
    tsk = task(task_name)
    logger.info('>> Loading data...')
    tsk.load(data_dir)
    logger.info('>> retrieving train/data instances...')
    train_X, train_y, test_X, test_y = utils.get_instances(tsk, split_train_dev=False)
    test_X_ref = test_X

    pipe = pipeline(pipeline_name)
  
    logger.info('>> training pipeline ' + pipeline_name)
    pipe.fit(train_X, train_y)
    
    logger.info('>> testing...')
    sys_y = pipe.predict(test_X)

    logger.info('>> evaluation...')
    logger.info(utils.eval(test_y, sys_y))
    
    #-------------------- important features--------------------------------
    #-------- shows top 5 features for hate classification -----------------
    
    #utils.important_hate_features(pipe.named_steps.frm, pipe.named_steps.clf)

    if print_predictions:
        logger.info('>> predictions')
        utils.print_all_predictions(test_X_ref, test_y, sys_y, logger)



#------------------------- identification of format and pipeline -------------------------

def task(name):
    if name == 'vua_format':
        return vf.VuaFormat()
    else:
        raise ValueError("task name is unknown. You can add a custom task in 'tasks'")

def pipeline(name):
    if name == 'svm_libsvc_counts_12':
        return pipelines.svm_libsvc_counts_12()    
    else:
        raise ValueError("pipeline name is unknown. You can add a custom pipeline in 'pipelines'")




