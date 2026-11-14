# User Manual - Diabetes Risk Check


The same guide is built into the app under **Help** (http://localhost:8080/manual.html).

This page explains, step by step, how to check your risk and how to read your result.

You do not need any technical knowledge.

## What this tool does

It asks 21 short questions about your health and habits and compares your answers with a large US health survey of about 250,000 people. It then tells you whether your risk of diabetes looks low, moderate or high, and which changes would lower it most.

> It is not a medical test and cannot tell you whether you have diabetes. Only a blood test can. Use the result as a reason to talk to a doctor, not as a diagnosis.

## Checking your risk

1. Open **Check my risk** from the top menu.

2. Answer the questions on each of the six screens. Tap **Yes** or **No**, pick an option from a list, or type a number.

3. Press **Next** to move on. If you missed a question it turns red and tells you what is missing.

4. On the *Your body* screen, type your height in centimetres and weight in kilograms. Your BMI is worked out for you. If you already know your BMI, choose **I know my BMI**.

5. You can go back at any time with **Back**. Your answers are kept.

6. On the last screen, press **See my result**.

![The first screen: about you](../screenshots/01_app_home.png)

![Answering questions with Yes / No buttons](../screenshots/02_app_form_health_conditions.png)

## Reading your result

- **Risk level** - low (green), moderate (amber) or high (red).
- **The coloured bar** - the black marker shows your score from 0 to 100. The colours show where low, moderate and high begin.
- **What would lower your risk most** - up to three changes, such as being more active or reaching a healthier weight, with how much each would lower your score. Things you cannot change, like age, are not listed.
- The score is a comparison with the survey, not the percentage chance that you have diabetes.

To try different answers, press **Change my answers**. To clear everything, press **Start again**.

![An example result: low risk, with the changes that would lower the score most](../screenshots/05_app_result.png)

## What to do next

- **Low:** keep up healthy habits and have regular check-ups.
- **Moderate:** mention it to your doctor and ask whether a blood sugar test makes sense.
- **High:** see a doctor soon and ask for a blood sugar (HbA1c) test.

## Your privacy

The tool never asks for your name, phone or email. To check that the model keeps working well, answers are stored in encrypted form without any personal identifiers.

## If something goes wrong

- **"We could not reach the risk service"** - check your internet connection and press the button again.
- **"The risk model is starting up or busy"** - wait a minute and try again.
- **A question stays red** - it still needs an answer, or a number is outside the allowed range (for example, days must be between 0 and 30).
- Error messages show a *reference* code. Give it to the support team so they can find the problem.

## The "How it works" page

This page is for the team that runs the tool. It shows how the model was built and checked, whether recent users look different from the people the model learned from, and links to the monitoring dashboards. You do not need it to check your risk.
