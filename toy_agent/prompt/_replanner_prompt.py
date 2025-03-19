replan_prompt_template: str = """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. \
If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. \
Only add steps to the plan that still NEED to be done. \
Do not return previously done steps as part of the plan.
"""
replan_prompt_template: str = """针对既定目标，制定一个简单的分步计划。 \
此计划应包括个人任务，如果正确执行，将得出正确答案。不要添加任何多余的步骤。 \
最终步骤的结果应该是最终答案。确保每一步都有所需的所有信息 - 不要跳过步骤。

你的目标是:
{input}

你的计划是:
{plan}

您目前已完成以下步骤:
{past_steps}

结合你的目标和已经完成的步骤，分析当前是否已经能够得出答案：
- 如果可以，请返回给用户。
- 如果不能，请相应地更新你的计划。如果不需要更多步骤，并且您可以返回给用户，那么请以该步骤进行响应。\
否则，填写计划。**只添加仍需要完成**的步骤到计划中。不要将已完成的步骤作为计划的一部分返回。
"""
