import copy
import operator

import attr
from ratsql.utils.logger_helper import get_logger
main_logger = get_logger("RAT-SQL")

@attr.s
class Hypothesis:
    inference_state = attr.ib()
    next_choices = attr.ib()
    score = attr.ib(default=0)

    choice_history = attr.ib(factory=list)
    score_history = attr.ib(factory=list)


def beam_search(model, orig_item, preproc_item, beam_size, max_steps):
    main_logger.info(f"orig_item={orig_item}")
    main_logger.info(f"preproc_item={preproc_item}")
    inference_state, next_choices = model.begin_inference(orig_item, preproc_item)
    m2c_align_mat = model.m2c_align_mat
    m2t_align_mat = model.m2t_align_mat
    beam = [Hypothesis(inference_state, next_choices)]
    finished = []

    for step in range(max_steps):
        # Check if all beams are finished
        if len(finished) == beam_size:
            break

        candidates = []

        # For each hypothesis, get possible expansions
        # Score each expansion
        for hyp in beam:
            candidates += [(hyp, choice, choice_score.item(),
                            hyp.score + choice_score.item())
                           for choice, choice_score in hyp.next_choices]

        # Keep the top K expansions
        candidates.sort(key=operator.itemgetter(3), reverse=True)
        candidates = candidates[:beam_size - len(finished)]

        # Create the new hypotheses from the expansions
        beam = []
        for hyp, choice, choice_score, cum_score in candidates:
            inference_state = hyp.inference_state.clone()
            next_choices = inference_state.step(choice)
            if next_choices is None:
                finished.append(Hypothesis(
                    inference_state,
                    None,
                    cum_score,
                    hyp.choice_history + [choice],
                    hyp.score_history + [choice_score]))
            else:
                beam.append(
                    Hypothesis(inference_state, next_choices, cum_score,
                               hyp.choice_history + [choice],
                               hyp.score_history + [choice_score]))

    finished.sort(key=operator.attrgetter('score'), reverse=True)
    return finished, m2c_align_mat, m2t_align_mat
