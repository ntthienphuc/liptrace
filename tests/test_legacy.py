import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from liptrace.audit import validate_splits, prediction_metrics
from liptrace.model import LipNetBackbone
from liptrace.training import greedy_decode, min_ctc_required_len, LipNetCsvDataset, read_video_gray_resize
from liptrace.runtime import load_recognizer


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ctc_blank_separates_repeated_characters(self):
        ids = [0,0,2,0,1,1,2]
        values = torch.nn.functional.one_hot(torch.tensor(ids),3).float().unsqueeze(1)
        self.assertEqual(greedy_decode(values,2,{0:'a',1:'b'}),['aab'])

    def test_ctc_minimum_alignment_length(self):
        self.assertEqual(min_ctc_required_len('book'),5)
        self.assertEqual(min_ctc_required_len('aaa'),5)

    def test_failed_video_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'bad.csv'
            pd.DataFrame([{'clip_path':str(Path(tmp)/'missing.mp4'),'text':'a'}]).to_csv(p,index=False)
            with self.assertRaises(RuntimeError):
                LipNetCsvDataset(str(p),{'a':0},6,16)[0]

    def test_video_packing_and_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'clip.avi'
            w = cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),10,(16,16))
            self.assertTrue(w.isOpened())
            for value in [0,100,240]:
                w.write(np.full((16,16,3),value,np.uint8))
            w.release()
            a = read_video_gray_resize(str(p),6,16)
            self.assertEqual(a.shape,(1,6,16,16))
            self.assertTrue(np.isfinite(a).all())
            np.testing.assert_equal(a[0,2],a[0,5])
            self.assertTrue(a.min()>=-1 and a.max()<=1)

    def test_split_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i in range(3):
                p=Path(tmp)/f'{i}.csv'
                pd.DataFrame([{'clip_path':f'{i}.mp4','text':'a','speaker_dir':'same'}]).to_csv(p,index=False)
                paths.append(p)
            with self.assertRaisesRegex(ValueError,'leakage'):
                validate_splits(paths)

    def test_empty_hypothesis_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pred.csv'
            pd.DataFrame([{'ref_text':'a','hyp_text':''},{'ref_text':'book','hyp_text':'bok'}]).to_csv(p,index=False)
            m=prediction_metrics(p)
            self.assertEqual(m['cer'],0.625)
            self.assertEqual(m['wer'],1)

    def test_checkpoint_contract_and_model_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.pt'
            cfg={'img_size':16,'max_frames':6,'rnn_units':4,'dropout':0.1}
            m=LipNetBackbone(2,**cfg)
            c={'model_state':m.state_dict(),'charset':['a'],'blank_index':1,'args':cfg}
            torch.save(c,p)
            loaded,_,_=load_recognizer(p)
            with torch.inference_mode():
                self.assertEqual(tuple(loaded(torch.zeros(1,1,6,16,16)).shape),(1,6,2))
            c['blank_index']=0
            torch.save(c,p)
            with self.assertRaisesRegex(ValueError,'blank mismatch'):
                load_recognizer(p)

    def test_unseen_training_character_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            files=[]
            for i,label in enumerate(['a','b','a']):
                clip=tmp/f'clip{i}.avi'
                w=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'MJPG'),10,(16,16))
                self.assertTrue(w.isOpened())
                for _ in range(6): w.write(np.zeros((16,16,3),np.uint8))
                w.release()
                p=tmp/f'split{i}.csv'
                pd.DataFrame([{'clip_path':str(clip),'text':label,'speaker_dir':f'group{i}'}]).to_csv(p,index=False)
                files.append(p)
            r=subprocess.run([sys.executable,'-m','liptrace.training','--train_csv',str(files[0]),'--val_csv',str(files[1]),'--test_csv',str(files[2]),'--out_dir',str(tmp/'run'),'--img_size','16','--max_frames','6','--epochs','1'],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn('unseen training characters',r.stderr)


if __name__ == '__main__':
    unittest.main()
