import torch
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import nltk
from nltk.util import ngrams
import traceback
import os
import json

class Evaluator:
    def __init__(self):
        try:
            self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.path_ ='../output_files/metrics_data_output/'
        except Exception as e:
            print(traceback.format_exc())


    def evaluate_bleu_rouge(self, candidates, references):
        try:
            bleu_score = corpus_bleu(candidates, [references]).score
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
            rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
            rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
            rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
            return bleu_score, rouge1, rouge2, rougeL
        except Exception as e:
            print(traceback.format_exc())

    def evaluate_bert_score(self, candidates, references):
        try:
            P, R, F1 = score(candidates, references, lang="en", model_type='bert-base-multilingual-cased')
            return P.mean().item(), R.mean().item(), F1.mean().item()
        except Exception as e:
            print(traceback.format_exc())

    def evaluate_perplexity(self, text):
        try:
            encodings = self.gpt2_tokenizer(text, return_tensors='pt')
            max_length = self.gpt2_model.config.n_positions
            stride = 512
            lls = []
            for i in range(0, encodings.input_ids.size(1), stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, encodings.input_ids.size(1))
                trg_len = end_loc - i
                input_ids = encodings.input_ids[:, begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100
                with torch.no_grad():
                    outputs = self.gpt2_model(input_ids, labels=target_ids)
                    log_likelihood = outputs[0] * trg_len
                lls.append(log_likelihood)
            ppl = torch.exp(torch.stack(lls).sum() / end_loc)
            return ppl.item()
        except Exception as e:
            print(traceback.format_exc())


    def evaluate_all(self, question, response, reference):
        try:
            candidates = [response]
            references = [reference]
            bleu, rouge1, rouge2, rougeL = self.evaluate_bleu_rouge(candidates, references)
            bert_p, bert_r, bert_f1 = self.evaluate_bert_score(candidates, references)
            perplexity = self.evaluate_perplexity(response)

            return {
                "BLEU": bleu,
                "ROUGE-1": rouge1,
                "ROUGE-2": rouge2,
                "ROUGE-L": rougeL,
                "BERT P": bert_p,
                "BERT R": bert_r,
                "BERT F1": bert_f1,
                "Perplexity": perplexity

            }
        except Exception as e:
            print(traceback.format_exc())

    def evaluate_average(self, list_of_metrics):
        try:
            
            s_Bl = s_R1= s_R2 =s_RL= s_B_P= s_B_R=s_B_F1 = s_P= 0

            len_=len(list_of_metrics)
            for metrics in range(0, len_):
                s_Bl=s_Bl+ list_of_metrics[metrics]['BLEU']
                s_R1=s_R1+ list_of_metrics[metrics]['ROUGE-1']
                s_R2=s_R2+ list_of_metrics[metrics]['ROUGE-2']
                s_RL=s_RL+ list_of_metrics[metrics]['ROUGE-L']
                s_B_P=s_B_P+ list_of_metrics[metrics]['BERT P']
                s_B_R=s_B_R+ list_of_metrics[metrics]['BERT R']
                s_B_F1=s_B_F1+ list_of_metrics[metrics]['BERT F1']
                s_P=s_P+ list_of_metrics[metrics]['Perplexity']
            


            a_Bl = float("{:.4f}".format(s_Bl/len_))
            a_R1 = float("{:.4f}".format(s_R1/len_))
            a_R2 = float("{:.4f}".format(s_R2/len_))
            a_RL = float("{:.4f}".format(s_RL/len_))
            a_BP = float("{:.4f}".format(s_B_P/len_))
            a_BR = float("{:.4f}".format(s_B_R/len_))
            a_BF1 = float("{:.4f}".format(s_B_F1/len_))
            a_P = float("{:.4f}".format(s_P/len_))

            Averaged_metrics= {
                'Average result': {
            'Blue':a_Bl,
            'ROUGE-1':a_R1,
            'ROUGE-2': a_R2,
            'ROUGE-L': a_RL,
            'BERT P' : a_BP,
            'BERT R' : a_BR,
            'BERT F1': a_BF1,
            'Perplexity': a_P
            }
            }
            print(Averaged_metrics)

            return Averaged_metrics

        except Exception as e:
            print(traceback.format_exc())