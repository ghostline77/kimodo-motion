"""Batch orchestration checks without loading model weights or a GPU."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from kimodo_test_batch import child


class BatchTest(unittest.TestCase):
    def test_independent_prompts_and_seeds_reuse_one_motion_model(self):
        calls, encodings, loads = [], [], []
        model = object()
        loading = types.ModuleType("kimodo.model.load_model")
        def load(*args, **kwargs):
            loads.append((args,kwargs))
            return model
        loading.load_model = load
        generate = types.SimpleNamespace()
        generate.get_generation_inputs = lambda args, fps: {"texts":["unused"],"num_frames":[120]}
        def generate_one():
            inputs=generate.get_generation_inputs(None,30)
            self.assertEqual(inputs["texts"],["unused"])
            self.assertIs(loading.load_model("same-checkpoint"),model)
            calls.append((sys.argv[1],sys.argv[-1]))
        generate.main = generate_one
        def bootstrap():
            encodings.extend(generate.get_generation_inputs(None,30)["texts"])
            generate.main()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            jobs=[]
            for i in range(2):
                output=root/str(i);output.mkdir()
                jobs.append({"id":str(i),"prompt":f"Prompt {i}.","output":str(output),
                             "arguments":[f"Prompt {i}.","--seed",str(101+i)]})
            modules={"kimodo.scripts":types.SimpleNamespace(generate=generate),
                     "kimodo.model.load_model":loading,
                     "runtime_bootstrap":types.SimpleNamespace(main=bootstrap)}
            with patch.dict(sys.modules,modules),patch.object(sys,"argv",["test"]):
                child({"jobs":jobs,"progress":str(root/"progress.json")})
            self.assertEqual(encodings,["Prompt 0.","Prompt 1."])
            self.assertEqual(calls,[("Prompt 0.","101"),("Prompt 1.","102")])
            self.assertEqual(len(loads),1)
            self.assertEqual(json.loads((root/"progress.json").read_text())["completed"],["0","1"])


if __name__=="__main__":
    unittest.main()
