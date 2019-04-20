import json

import tensorflow as tf

import model
from model import default_hparams

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )

def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])

        def body(past, prev, output):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
            ]

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

    return tokens


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="path to the model checkpoint")
    parser.add_argument('vocab', type=str, help="path to the char_to_idx mapping json file")
    parser.add_argument('--hparams', type=str, help="path to the json-stored model hyperparameters")
    parser.add_argument('--length', type=int, help="character length of text samples")
    parser.add_argument('--batch_size', type=int, default=1, help="number of samples to be sampled in one run")
    parser.add_argument('--temperature', type=float, default=1., help="degree of ceratinty when generating samples. 0 means deterministic")
    parser.add_argument('--top_k', type=int, default=0, help="how much characters to consider at each sampling step. 1 means most deterministic")
    parser.add_argument('--n_samples', type=int, default=1, help="number of samples to generate")
    parser.add_argument('--seed', type=int, default=144, help="random seed for reproducibility")
    args = parser.parse_args()

    with open(args.vocab, 'r', encoding='utf-8') as fp:
        char_to_idx = json.load(fp)
        idx_to_char = {v: k for k, v in char_to_idx.items()}
    
    hparams = default_hparams()
    if args.hparams is not None:
        with open(args.hparams, 'r') as hf:
         hparams.parse_json(hf.read())

    length = args.length if args.length is not None else hparams.n_ctx // 2

    # build graph
    context = tf.placeholder(tf.int32, [args.batch_size, None])
    tf.set_random_seed(args.seed)
    output = sample_sequence(
        hparams=hparams,
        length=length,
        context=context,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_k=args.top_k
    )

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, args.model)

        primed_text = 'Haba'

        context_tokens = [char_to_idx[x] for x in primed_text]
        generated = 0
        for _ in range(args.n_samples // args.batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(args.batch_size)]
            })#[:, len(context_tokens):]
            for i in range(args.batch_size):
                generated += 1
                text = [idx_to_char[x] for x in out[i]]
                text = ''.join(text)
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)



if __name__ == "__main__":
    main()