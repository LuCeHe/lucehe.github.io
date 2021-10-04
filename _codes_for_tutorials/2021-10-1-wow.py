import os, json, shutil
import tensorflow as tf

from GenericTools.KerasTools.esoteric_layers import AddLossLayer, ReplaceColumn
from GenericTools.KerasTools.esoteric_models.transformer import GPT
from GenericTools.KerasTools.huggingface_tools import HF_ModelUpgrade

tf.compat.v1.enable_eager_execution()

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint

from GenericTools.KerasTools.advanced_losses import *
from GenericTools.KerasTools.esoteric_models.wizard_of_wikipedia import metrics_wow, switch_external_knowledge, \
    tf_ContextKnowledgeEncoder, tf_ContextKnowledgeDecoder, UniversalSentenceEmbedding
from GenericTools.KerasTools.esoteric_models.wizard_of_wikipedia import EndToEndModel
from GenericTools.KerasTools.esoteric_optimizers.optimizer_selection import get_optimizer
from GenericTools.KerasTools.esoteric_tasks.wizard_of_wikipedia import WikipediaWizardGenerator
from GenericTools.StayOrganizedTools.VeryCustomSacred import CustomExperiment, ChooseGPU
from GenericTools.KerasTools.esoteric_callbacks import *
from GenericTools.KerasTools.plot_tools import plot_history
from GenericTools.StayOrganizedTools.utils import setReproducible, str2val

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, 'data', 'wizard_of_wikipedia'))
os.makedirs(DATAPATH, exist_ok=True)

ex = CustomExperiment('dialogue', base_dir=CDIR, seed=0)
models_dict = {
    'E2E': EndToEndModel,
}


@ex.config
def config():
    show_dialogue = False
    make_model = True
    maxlen = 12
    max_knowledge = 32
    batch_size = 4
    n_dialogues = 'full'  # -1 100 random
    epochs = 1
    steps_per_epoch = None
    stop_time = 500  # 72000 = 20h, 54000 = 15h
    seed = 4
    model_name = 'E2E'
    vocab_size = 34883 if not 'Pretrained' in model_name else 50257

    # comments = 'encoder_maxlen:128-decoder_maxlen:12'
    comments = ''
    tests = ['on_data', 'max', 'beam', 'evaluations']
    # tests = ['beam']
    load_model_path = None
    # load_model_path = r'C:\Users\PlasticDiscobolus\work\ariel_tests\experiments\2021-09-26--15-58-52--8279-dialogue_\trained_models\model_weights.h5'


# longest dialogue: 23 utterances in train
# vocabulary original WoW network, with BPE, 34883 subwords. Mine 29999

load_m = lambda x, mask_value, num_classes: tf.keras.models.load_model(
    x, custom_objects=
    {'tf_ContextKnowledgeEncoder': tf_ContextKnowledgeEncoder, 'tf_ContextKnowledgeDecoder': tf_ContextKnowledgeDecoder,
     'masked_xent': masked_sparse_crossentropy(mask_value), 'masked_perplexity': masked_sparse_perplexity(mask_value),
     'sparse_f1_on_max': sparse_f1_on_max(num_classes), 'masked_f1_on_max': masked_f1_on_max(num_classes, mask_value),
     'UniversalSentenceEmbedding': UniversalSentenceEmbedding, 'AddLossLayer': AddLossLayer,
     'sparse_perplexity': sparse_perplexity,      }
)


@ex.automain
def main(show_dialogue, make_model, maxlen, max_knowledge, vocab_size, batch_size,
         n_dialogues, epochs, stop_time, steps_per_epoch, seed, model_name, _log, comments, tests, load_model_path):
    ChooseGPU(None)
    exp_dir = os.path.join(CDIR, ex.observers[0].basedir)
    setReproducible(seed)

    print(json.dumps(ex.configurations[-1](), indent=4))
    print(json.dumps(ex.observers[0].updated_config, indent=4))
    print('experiment: ', exp_dir)

    if show_dialogue:
        data_json = os.path.join(DATAPATH, 'train.json')

        with open(data_json) as f:
            data = json.load(f)

        print(data[1].keys())
        for d in data[1]['dialog']:
            print('-' * 39)
            print(d.keys())
            print(d['speaker'])
            print(d['text'])
            if 'checked_sentence' in d.keys():
                print(d['checked_sentence'])
            rp = [' '.join(list(l.values())[0]) for l in d['retrieved_passages']]
            print(json.dumps(rp, indent=4, sort_keys=True))
            print(len(rp[0]))

    # json to gpt2 indices

    if make_model:
        other_dir = os.path.join(exp_dir, 'other_outputs')
        images_dir = os.path.join(exp_dir, 'images')

        path_best_model = os.path.join(exp_dir, 'trained_models', 'model_weights.h5')

        history_path = other_dir + '/log.csv'
        callbacks = [
            # IndividualWeightsTensorBoard(log_dir=other_dir, histogram_freq=5),
            ModelCheckpoint(path_best_model, monitor='val_masked_perplexity', verbose=1, save_best_only=True,
                            mode='min'),
            LearningRateLogger(),
            tf.keras.callbacks.CSVLogger(history_path),
            TimeStopping(stop_time, 1),
        ]

        tokenizer_choice = 'bpe' if not 'Pretrained' in model_name else 'gpt2'
        encoder_maxlen = str2val(comments, 'encoder_maxlen', int, default=maxlen, split_symbol='-')
        decoder_maxlen = str2val(comments, 'decoder_maxlen', int, default=maxlen, split_symbol='-')
        gen_train = WikipediaWizardGenerator(
            data_path=DATAPATH, n_dialogues=n_dialogues, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
            encoder_maxlen=encoder_maxlen, decoder_maxlen=decoder_maxlen, epochs=epochs,
            tokenizer_choice=tokenizer_choice, data_split='train')
        gen_val = WikipediaWizardGenerator(
            data_path=DATAPATH, n_dialogues=n_dialogues, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
            encoder_maxlen=encoder_maxlen, decoder_maxlen=decoder_maxlen, tokenizer_choice=tokenizer_choice,
            data_split='valid_random_split')

        tokenizer = gen_train.tokenizer
        pad_idx = gen_train.pad_idx

        rate = str2val(comments, 'dropout', float, default=.1, split_symbol='-')
        model = models_dict[model_name](
            max_knowledge=max_knowledge, input_vocab_size=vocab_size, target_vocab_size=vocab_size, pad_idx=pad_idx,
            encoder_maxlen=encoder_maxlen, decoder_maxlen=decoder_maxlen, rate=rate)

        optimizer = get_optimizer(
            'Adam', .0005, lr_schedule='cosine_no_restarts',
            total_steps=gen_train.epochs, clipnorm=.1,
            warmup_steps=gen_train.steps_per_epoch
        )
        model.compile(
            optimizer,
            sparse_perplexity,  # masked_sparse_crossentropy(mask_value=pad_idx),
            metrics=metrics_wow(num_classes=vocab_size, mask_value=pad_idx)
        )

        if load_model_path is None:
            model.fit(gen_train, epochs=gen_train.epochs, validation_data=gen_val, callbacks=callbacks)
            if gen_train.epochs > 1:
                # model.load_weights(path_best_model)
                model = load_m(path_best_model, mask_value=pad_idx, num_classes=vocab_size)

        else:
            model = load_m(load_model_path, mask_value=pad_idx, num_classes=vocab_size)

        if os.path.exists(path_best_model):
            os.remove(path_best_model)
        model.summary()
        # Re-evaluate the model

        results = {}
        if 'evaluations' in tests:
            for split_name in ['valid']:
                for random_topic in ['random', 'topic']:
                    for knowledge_switch in ['on', 'off']:
                        try:
                            data_split = '{}_{}'.format(split_name, random_topic)
                            print('data split: {}, knowledge_switch: {}'.format(data_split, knowledge_switch))
                            gen = WikipediaWizardGenerator(
                                data_path=DATAPATH, n_dialogues=n_dialogues, batch_size=batch_size,
                                steps_per_epoch=steps_per_epoch,
                                encoder_maxlen=encoder_maxlen, decoder_maxlen=decoder_maxlen,
                                tokenizer_choice=tokenizer_choice,
                                data_split=data_split + '_split')
                            switch_external_knowledge(model, state=knowledge_switch)
                            evaluation = model.evaluate(gen, return_dict=True)
                            results.update(
                                {k + '_' + data_split + '_' + knowledge_switch: v for k, v in evaluation.items()})
                        except Exception as e:
                            print(e)

            print(results)
        if gen_train.epochs > 0 and load_model_path is None:
            h = pd.read_csv(history_path)
            history_dict = {k: h[k].tolist() for k in h.columns.tolist()}
            plot_history(history_dict, os.path.join(images_dir, 'history.png'), gen_train.epochs)

        all_sentences = []

        base_input_batch = gen_val.__getitem__()[0]
        try:
            if 'on_data' in tests:
                all_sentences.append('\n\nTokenizer on data:')
                gen_val.on_epoch_end()
                batch = gen_val.data_generation()
                for sample in batch['targets']:
                    decoded = tokenizer.decode(sample)
                    all_sentences.append('\n' + decoded)

                input_batch = base_input_batch
                for knowledge_switch in ['on', 'off']:
                    switch_external_knowledge(model, state=knowledge_switch)
                    all_sentences.append(
                        '\n\nTokenizer on max predictions, knowledge switch {}:'.format(knowledge_switch))

                    prediction = model.predict(input_batch)
                    max_prediction = tf.argmax(prediction, -1)
                    for mdl_sample, tgt_sample in zip(max_prediction, input_batch[-1]):
                        mdl_output = tokenizer.decode(mdl_sample)
                        tgt_output = tokenizer.decode(tgt_sample)
                        all_sentences.append('\nmodel output:  {}'.format(mdl_output))
                        all_sentences.append('\ntarget output: {}'.format(tgt_output))
        except Exception as e:
            print(e)

        try:
            if 'max' in tests:
                for knowledge_switch in ['on', 'off']:
                    switch_external_knowledge(model, state=knowledge_switch)
                    all_sentences.append('\n\nGenerations, knowledge switch {}:'.format(knowledge_switch))

                    input_batch = base_input_batch
                    # generated_sentence = input_batch[-1][:, 0][..., None]
                    generated_sentence = np.repeat(np.array([gen_train.start_idx] * batch_size)[..., None], maxlen, -1)
                    input_batch[-1] = generated_sentence
                    for i in range(maxlen - 1):
                        prediction = model.predict(input_batch)
                        new_token = np.argmax(prediction[:, i], -1)
                        generated_sentence[:, i + 1] = new_token
                        input_batch[-1] = generated_sentence

                    for sample in generated_sentence:
                        generated_sentence = tokenizer.decode(sample)
                        all_sentences.append('\nmodel generation:  {}'.format(generated_sentence))
        except Exception as e:
            print(e)

        try:
            if 'beam' in tests:
                for knowledge_switch in ['on', 'off']:
                    switch_external_knowledge(model, state=knowledge_switch)
                    all_sentences.append(
                        '\n\nGenerations, HuggingFace beam-search, knowledge switch {}:'.format(knowledge_switch))

                    input_batch = base_input_batch
                    hf_model = HF_ModelUpgrade(model, input_batch[:-1], gen_val.start_idx, pad_idx, pad_idx, vocab_size)

                    generated_sentence = hf_model.generate(
                        input_ids=tf.constant(input_batch[-1][:, 0][..., None]), num_beams=3, num_return_sequences=1,
                        do_sample=False, max_length=decoder_maxlen, min_length=3, fixed_length_input=True
                    )
                    for sample in generated_sentence:
                        generated_sentence = tokenizer.decode(sample)
                        all_sentences.append('\nmodel generation:  {}'.format(generated_sentence))
        except Exception as e:
            print(e)

        text_path = os.path.join(exp_dir, 'text', 'sentences.txt')
        with open(text_path, 'w', encoding="utf-8") as f:
            for sentence in all_sentences:
                _log.info(sentence)
                f.write(sentence)

    results_filename = os.path.join(other_dir, 'results.json')
    json.dump(results, open(results_filename, "w"))
    _log.info('DONE!')
    shutil.make_archive(exp_dir, 'zip', exp_dir)
