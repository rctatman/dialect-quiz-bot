## happy path
* greet
    - utter_start_quiz
* affirm
    - elicitation_form
    - form{"name": "elicitation_form"}
    - form{"name": null}
    - utter_slots_values
    - detect_dialect
    - utter_dialect_value
* thankyou
    - utter_noworries

## no quiz
* greet
    - utter_start_quiz
* deny
    - utter_noworries

## bot challenge
* bot_challenge
  - utter_iamabot

## toxic language
* toxic_language
  - pause_conversation

## interrupted form stop
* greet
    - utter_start_quiz
* affirm
    - elicitation_form
    - form{"name": "elicitation_form"}
* bot_challenge
    - utter_iamabot
    - utter_ask_continue
* deny
    - action_deactivate_form
    - form{"name": null}
    - utter_goodbye

## interrupted form continue
* greet
    - utter_start_quiz
* affirm
    - elicitation_form
    - form{"name": "elicitation_form"}
* bot_challenge
    - utter_iamabot
    - utter_ask_continue
* affirm
    - elicitation_form
    - form{"name": null}
    - utter_slots_values
    - detect_dialect
    - utter_dialect_value
* thankyou
    - utter_noworries
    - utter_goodbye

## interrupted form chitchat
* greet
    - utter_start_quiz
* affirm
    - elicitation_form
    - form{"name": "elicitation_form"}
* chitchat
    - utter_chitchat
    - elicitation_form
    - form{"name": null}
    - utter_slots_values
    - detect_dialect
    - utter_dialect_value
* thankyou
    - utter_noworries
    - utter_goodbye
