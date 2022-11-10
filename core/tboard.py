import tensorflow as tf


def draw_graph(model, dataset, writer, logdir=''):
    '''Decorator that reports store fn graph.'''

    @tf.function
    def fn(x):
        x = model(x)

    tf.summary.trace_on(graph=True, profiler=False)
    fn(dataset)
    with writer.as_default():
        tf.summary.trace_export(
            name='model',
            step=0,
            profiler_outdir=logdir)


def save_scalar(writer, value, step, name=''):
    with writer.as_default():
        tf.summary.scalar(name, value.result(), step=step)
