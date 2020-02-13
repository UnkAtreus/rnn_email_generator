import tensorflow as tf
import numpy as np
import my_txtutils

ALPHASIZE = my_txtutils.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512

shakespeare = "pretrained/rnn_test_minibatchseq_1477670023-174939000"
python = "pretrained/rnn_test_minibatchseq_1477834023-138609000"
harrypotter = "pretrained/"
mail = "pretrained/rnn_train_1581472865-10500000"

#author = shakespeare
#author = python
#author = harrypotter
author = mail

ncnt = 0
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('pretrained/rnn_train_1581472865-10500000.meta')
    new_saver.restore(sess, author)
    x = my_txtutils.convert_from_alphabet(ord("K"))
    x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    # initial values
    y = x
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    for i in range(1000000000):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})
        c = my_txtutils.sample_from_probabilities(yo, topn=2)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = chr(my_txtutils.convert_to_alphabet(c))
        print(c, end="")

        if c == '\n':
            ncnt = 0
        else:
            ncnt += 1
        if ncnt == 100:
            print("")
            ncnt = 0
