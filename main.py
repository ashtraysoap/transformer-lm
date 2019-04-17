import tensorflow as tf
from tensorflow.contrib.training import HParams

import datagen as gen
import model
from sample import sample_sequence

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=256,
        n_head=4,
        n_layer=3,
        max_grad_norm=1,
        lr=6.25e-5,
        lr_warmup=0.002,
        l2=0.01,
        vector_l2='store_true',
        lr_schedule='warmup_linear',
        b1=0.9,
        b2=0.999,
        e=1e-8,
        opt='adam',
        batch_size=8,
        n_epochs=10,
        stride=8,
        sample_every=1000
    )

def main():
    hp = default_hparams()    
    fname = 'n_92105_filt.txt'
    cti = gen.make_char_to_idx(fname)
    itc = {v: k for k, v in cti.items()}
    
    print("Character to Index dictionary:", cti)
    hp.n_vocab = len(cti)
    print("Number of elements in the vocabulary:", hp.n_vocab)
    
    batch_size = hp.batch_size
    total_chars = gen.get_char_count(fname)
    total_updates = ((total_chars - (hp.n_ctx + 1)) // hp.stride + 1) // batch_size * hp.n_epochs
    hp.n_updates_total = total_updates

    g = gen.data_iterator(fname, cti, buffer=65536, context=hp.n_ctx, batch=batch_size, stride=8)

    context = tf.placeholder(tf.int32, [batch_size, None])
    labels = tf.placeholder(tf.int32, [batch_size, None])

    # res = model.model(hp, context)
    # print(res)
    
    # logits = res['logits']
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    # loss = tf.reduce_mean(loss, axis=None)
    # train_ops = tf.train.AdamOptimizer().minimize(loss)

    loss, train_ops = model.get_train_ops(hp, context, labels, past=None)

    # sample every `sample_steps`
    sample_steps = hp.sample_every
    steps = 0

    output = sample_sequence(hparams=hp,
                            length=hp.n_ctx // 2,
                            context=context,
                            batch_size=batch_size,
                            temperature=1,
                            top_k=5)
    
    primed_text = "Keď stál o mnoho rokov neskôr pred popravnou čatou, spomenul si "
    primed_text = [cti[c] for c in primed_text]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(hp.n_epochs):
            print("Epoch", e + 1)
            for batch in g:
                l, _ = sess.run((loss, train_ops),feed_dict={context: batch['features'],
                                                            labels: batch['labels']})
                print(l)
            
                steps += 1
                if steps % sample_steps == 0:
                    # sample model
                    print("================= Sampling =================")
                    out = sess.run(output, feed_dict={context: batch_size * [primed_text]})
                    for i in range(out.shape[0]):
                        text = ''.join([itc[x] for x in out[i]])
                        print(i)
                        print(text)

        print("End of training, final sample:")
        out = sess.run(output, feed_dict={context: batch_size * [primed_text]})
        for i in range(out.shape[0]):
            text = ''.join([itc[x] for x in out[i]])
            print(i)
            print(text)


if __name__ == "__main__":
    main()