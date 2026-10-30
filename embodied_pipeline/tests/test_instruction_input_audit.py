"""Offline tests: input revision must not mutate original evidence or context."""
import copy
import unittest
from embodied_pipeline.data_collection.instruction_input_audit import revise
from embodied_pipeline.data_collection.instruction_revision_specs import SYNTAX,RACK,TAPE,PILLOW,CONTAINER,ROOM_GAPS

class InstructionRevisionTests(unittest.TestCase):
    def task(self,rid,instruction,history):
        context={'source':'scenario_assumption_not_observation','user_location':{'room_name':'office_A','near_scene_element_id':101},'dialogue_history':[{'speaker':'user','text':history}]}
        return dict(record_id=rid,instruction=instruction,reading_aid_zh='原任务',draft_version='original',conversation_context=context,
                    public_spatial_context={'robot_location':{'status':'unknown'}},
                    model_input={'instruction':instruction,'conversation_context':copy.deepcopy(context)})

    def test_selection_is_disjoint_and_34_ready(self):
        ids=set(SYNTAX)|RACK|TAPE|PILLOW|CONTAINER
        self.assertEqual(len(ids),39)
        self.assertEqual(len(ids-set(ROOM_GAPS)),34)
        self.assertEqual([len([r for r in ids-set(ROOM_GAPS) if r.startswith(s+'_')]) for s in ['scene200','scene202','d3_619']],[13,10,11])

    def test_syntax_does_not_replace_prior_dialogue(self):
        t=self.task('scene202_10000050','Old instruction','I was considering the black notebook, but do not start yet.')
        old=copy.deepcopy(t);new=revise(t)
        self.assertEqual(t,old)
        self.assertEqual(new['conversation_context'],t['conversation_context'])
        self.assertNotEqual(new['instruction'],t['instruction'])
        self.assertEqual(new['model_input']['instruction'],new['instruction'])
        self.assertEqual(new['public_spatial_context'],t['public_spatial_context'])

    def test_same_item_retains_contextual_nature(self):
        t=self.task('scene202_10000016','Move that same item there.','The item I mean is the black-and-red tape dispenser. I have not asked you to move it yet.')
        new=revise(t)
        self.assertEqual(new['instruction'],t['instruction'])
        text=new['conversation_context']['dialogue_history'][0]['text']
        self.assertIn('currently supports the white tape roll',text)
        self.assertEqual(new['model_input']['context_messages'][0]['content'],text)
        self.assertNotIn('currently supports',t['conversation_context']['dialogue_history'][0]['text'])

if __name__=='__main__':unittest.main()
