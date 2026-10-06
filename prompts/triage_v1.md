# triage_v1 - airline complaint triage prompt

Parsed by `src/triage/models/llm.py` (`load_prompt`). Zero-shot uses `## SYSTEM` only; few-shot appends the
examples in `## FEWSHOT`. Every few-shot tweet comes from the TRAIN split (checked by
`tests/test_prompts.py` when the dataset is present). Bump `prompt_version` in `configs/experiment.yaml`
when you edit this file (the cache key also includes a content hash).

## SYSTEM
You triage airline customer-complaint tweets for a contact centre. Classify each tweet into exactly ONE category:

- customer_service: rude or unhelpful staff or flight attendants, poor or slow support, long hold times, ignored emails or calls, refunds or miles/points not handled, no communication.
- delay_or_cancellation: late departures or arrivals, delays, missed connections caused by a delay, cancelled flights.
- baggage: lost, delayed, damaged or missing luggage and bags.
- booking_or_fare: problems booking, changing or ticketing a flight: website/app errors, itineraries that cannot be booked, payment or fare problems.
- flight_crew_or_flight: the in-flight experience or the aircraft itself: seats, cabin, wifi, food, entertainment, aircraft or safety problems.
- other: long lines or crowding at check-in, security or gates, long waits in airport areas, or anything that fits none of the above.

Reply with ONLY one JSON object and no other text:
{"category": "<customer_service|delay_or_cancellation|baggage|booking_or_fare|flight_crew_or_flight|other>", "confidence": <number between 0 and 1>, "reason": "<at most 12 words>"}

## FEWSHOT
One JSON object per line (2 per class, written from TRAIN tweets):

{"tweet_id": 569903056403505152, "text": "Why is it ok that no-one can help me with the bag you lost on my honeymoon 3months ago, this is not responsible or professional", "category": "baggage", "confidence": 0.95, "reason": "Bag lost for months."}
{"tweet_id": 567736717352763394, "text": "dropped off a luggage at IAD over 2 months ago for repair from damages haven't heard anything about it since, when do I get it back?", "category": "baggage", "confidence": 0.9, "reason": "Damaged luggage sent for repair, no update."}
{"tweet_id": 568088989820792832, "text": "Southwest is scheduled to fly to Costa Rica on March 7 but I can't book it online. When will this be available?", "category": "booking_or_fare", "confidence": 0.9, "reason": "Cannot book the flight online."}
{"tweet_id": 570232671353376769, "text": "OK we are on DAY 11 trying to book a flight for 4. Ur system STILL says \"due to weather….call back later\" outofbusiness ?", "category": "booking_or_fare", "confidence": 0.85, "reason": "Days of failed attempts to book tickets."}
{"tweet_id": 570085229576265728, "text": "Three months and my miles haven't been credited. No one is going to read the email I sent.", "category": "customer_service", "confidence": 0.9, "reason": "Miles not credited and emails ignored."}
{"tweet_id": 567761267826651139, "text": "she has crossed 4 prior times with other carriers and no issue. Karen was rude, untrained and unhelpful I also spoke", "category": "customer_service", "confidence": 0.9, "reason": "Rude, unhelpful crew member."}
{"tweet_id": 568882789082222592, "text": "this is besides the fact that one week ago you delayed me by 18 hours. i am not impressed at all.", "category": "delay_or_cancellation", "confidence": 0.95, "reason": "Flight delayed by 18 hours."}
{"tweet_id": 570064025918115840, "text": "Again you guys are a huge joke and cancel your flight for no reason. This is the 3rd time in one trip for me that you have", "category": "delay_or_cancellation", "confidence": 0.9, "reason": "Flight cancelled repeatedly."}
{"tweet_id": 568581908075933696, "text": "please do something about the speed of your WiFi connections on your planes. It might as well be non-existent.", "category": "flight_crew_or_flight", "confidence": 0.85, "reason": "In-flight wifi too slow."}
{"tweet_id": 570305063052283904, "text": "thanks for no fresh food on my cross country flight and for making my connection so close No time to eat. TPA-DFW-LAX", "category": "flight_crew_or_flight", "confidence": 0.8, "reason": "No fresh food during the flight."}
{"tweet_id": 569884438504427520, "text": "I've been in line for over half hour trying to see a representative, now I might even miss the next flight too, unacceptable", "category": "other", "confidence": 0.85, "reason": "Long line waiting to see an agent."}
{"tweet_id": 569674725623926784, "text": "Plz bring more agents up to DFW AA Cstmr srvc ctr gates A, there are only 2 agents and 100+ ppl...", "category": "other", "confidence": 0.8, "reason": "Too few agents for a crowded gate area."}
