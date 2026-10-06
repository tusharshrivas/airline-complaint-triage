# Error analysis

20 misclassified test examples per model, allocated to the most common confusions (seed 42). Tweets are shown after cleaning. Source labels come from a single crowd-annotation pass, so some 'errors' are annotation noise or genuinely ambiguous tweets.

## TF-IDF + Logistic Regression

341 errors out of 1186 test items (28.8%). Showing 20 examples, allocated to confusion pairs in proportion to how common they are.

| true -> predicted | errors |
|---|---|
| customer_service -> delay_or_cancellation | 47 |
| delay_or_cancellation -> customer_service | 42 |
| customer_service -> booking_or_fare | 35 |
| customer_service -> flight_crew_or_flight | 22 |
| booking_or_fare -> customer_service | 21 |
| customer_service -> baggage | 19 |
| flight_crew_or_flight -> customer_service | 18 |
| baggage -> customer_service | 16 |

| # | tweet_id | true | predicted | conf | tweet (cleaned) | LLM reason |
|---|---|---|---|---|---|---|
| 1 | 570298036938772483 | customer_service | delay_or_cancellation | 0.30 | Why haven't you issued a travel advisory for Charlotte Wednesday night and Thursday? 4-6 inches of snow. |  |
| 2 | 569636885863251969 | customer_service | delay_or_cancellation | 0.57 | Two hour wait for EXPs as I sit on a JFK PHX flight because US computers are down. Any shot at an LAX flight? |  |
| 3 | 569682325778231297 | customer_service | delay_or_cancellation | 0.30 | please start flying to Huntsville so I never have to fly American Airlines again |  |
| 4 | 568835810532794368 | delay_or_cancellation | customer_service | 0.40 | cleaning a regional jet takes an hour? |  |
| 5 | 568954558589702144 | delay_or_cancellation | customer_service | 0.66 | I will send an email with details later. Thank you for responding. |  |
| 6 | 569627346166284288 | customer_service | booking_or_fare | 0.38 | does anyone actually work at the dividend miles department? |  |
| 7 | 568224769872502784 | customer_service | booking_or_fare | 0.95 | yeah you guys just told me to call the website that I booked from. I booked on your website. UNACCEPTABLE |  |
| 8 | 568422454609219584 | customer_service | flight_crew_or_flight | 0.54 | they had to turn the seat cushions over and clean the area. Please explain this issue? |  |
| 9 | 567739553251725312 | booking_or_fare | customer_service | 0.50 | Twitter says I can't DM someone unless they follow me. Can follows my twitter? thanks you. |  |
| 10 | 567729135733465088 | customer_service | baggage | 0.76 | thank you for blowing my vacation. Couldn't get me anywhere today to make my reservation and also lost 2 bags of mine! |  |
| 11 | 568181167054196736 | flight_crew_or_flight | customer_service | 0.35 | took this one just for you. Not a window. Also not fun if you get motion sick. |  |
| 12 | 569576418855620608 | baggage | customer_service | 0.32 | no consistency, Denver agents say no standby with checked baggage citing FAA. Your website policy says otherwise. time2switch |  |
| 13 | 569616554922418177 | customer_service | other | 0.45 | there was also not one single person at the counter answering questions for our plane full of confused people. No staff at all. |  |
| 14 | 570049295040307200 | flight_crew_or_flight | delay_or_cancellation | 0.23 | UA 746. Pacific Rim and Date Night cut out. Not constantly or randomly, but one spot, repeatably. |  |
| 15 | 570187686213918721 | delay_or_cancellation | flight_crew_or_flight | 0.43 | so great AA1103 sitting for an hour first technical problems now what? |  |
| 16 | 569257540023951360 | delay_or_cancellation | other | 0.40 | we are sitting on the runway for 2 hours! It is ridiculous!! |  |
| 17 | 569847788986462209 | delay_or_cancellation | booking_or_fare | 0.27 | wasn't just a delay. Your counter wouldn't take a valid CAC card as a valid ID which is needed for a TSA precheck on pass |  |
| 18 | 569675799122591746 | baggage | delay_or_cancellation | 0.26 | To add, I have to get up to go to work tomorrow am. I don't like having to wait up until 12 or later. CS sucks. |  |
| 19 | 570066450796257280 | delay_or_cancellation | baggage | 0.32 | Delay DEN-CLE because they have to manually enter baggage tags? Really? Worst cust service day for this 1ker. friendlyskies?? |  |
| 20 | 568223950926446593 | other | delay_or_cancellation | 0.69 | still waiting! Captain reports he's called 6 times to get ground crew....and still sitting on runway. evenlater |  |

## TF-IDF + XGBoost

338 errors out of 1186 test items (28.5%). Showing 20 examples, allocated to confusion pairs in proportion to how common they are.

| true -> predicted | errors |
|---|---|
| customer_service -> delay_or_cancellation | 49 |
| delay_or_cancellation -> customer_service | 44 |
| customer_service -> flight_crew_or_flight | 30 |
| booking_or_fare -> customer_service | 27 |
| customer_service -> baggage | 26 |
| customer_service -> booking_or_fare | 23 |
| flight_crew_or_flight -> customer_service | 18 |
| delay_or_cancellation -> flight_crew_or_flight | 15 |

| # | tweet_id | true | predicted | conf | tweet (cleaned) | LLM reason |
|---|---|---|---|---|---|---|
| 1 | 569507349548957696 | customer_service | delay_or_cancellation | 0.36 | - on hold 45 minutes trying to rebook cancelled flight. Really? |  |
| 2 | 569671413671591937 | customer_service | delay_or_cancellation | 0.80 | your service at PHL is abysmal. An hour on the runway waiting for a gate, no information anywhere, missed my connection abysmal! |  |
| 3 | 570265133638942720 | customer_service | delay_or_cancellation | 0.65 | u have a lot of pissed off hungry tired people stuck at midway 4 over 2 hours after scheduled departure. This is inexcusable |  |
| 4 | 569897060717105152 | delay_or_cancellation | customer_service | 0.54 | I was told I had a 20minute wait time after waiting hours. And an hour has gone by. This is ridiculous |  |
| 5 | 569618829627666432 | delay_or_cancellation | customer_service | 0.44 | I have to go back home! I have to use what the company has available. But it's unfair to stay more than 24 h traveling |  |
| 6 | 569721575492194304 | delay_or_cancellation | customer_service | 0.27 | not anymore. |  |
| 7 | 568490310524936192 | customer_service | flight_crew_or_flight | 0.54 | Capital One and I explained the false fraud alert. Why did the Jet Blue representative issue me a new tkt if it wasn't resolved? |  |
| 8 | 569643287918845952 | customer_service | flight_crew_or_flight | 0.37 | If seats aren't guaranteed why do we pay for them? when I called the rep said some other people booked our seats. |  |
| 9 | 569624865558245376 | booking_or_fare | customer_service | 0.93 | LGA 2 Nashville cancelled phone center no help. Fabulous staff at gate D4 helped-2 young men handled crowd well. |  |
| 10 | 568963705439789056 | booking_or_fare | customer_service | 0.33 | express my disappointment in your reservation department and the treatment of a Sailor dealing with a death in the family |  |
| 11 | 570008958691364864 | customer_service | baggage | 0.59 | your DCA baggage claim employees should realize a please, a thank you and an apology can go a long way customerservice DCA |  |
| 12 | 568862914154770432 | customer_service | baggage | 0.60 | MIA-EWR 384 😄😄😄 excellent crew. EWR-IAD 3589 😡😡😡 No crew to load bags - waiting w/ door open freezing. 20 mins past departure. |  |
| 13 | 569236251129163777 | customer_service | booking_or_fare | 0.80 | your mobile site is broken, shows "{{header.elevateUser.numOfPointsAvailable \|\| '0' \| number}} Points", won't let me checkin |  |
| 14 | 567808893464899584 | flight_crew_or_flight | customer_service | 0.48 | if ur going 2 charge $20 for wi-fi make sure it works. Brutal DialUp b a 100 by the time things load. TheEnd GoodDay |  |
| 15 | 568917611565592577 | delay_or_cancellation | flight_crew_or_flight | 0.54 | Don't ask me to be patient without offering something in return. |  |
| 16 | 570013961808187393 | baggage | customer_service | 0.58 | i have and you have not been able to locate them, its insane. i trust you and PAY you to look after my things and you lost them. |  |
| 17 | 569995212296200192 | customer_service | other | 0.73 | already spoke to that line, unwilling to help - really poor support hitawall |  |
| 18 | 568626612314308608 | flight_crew_or_flight | delay_or_cancellation | 0.41 | why do you have every channel but How are and I supposed to watch Scandal? But free FlyFi is sweet! |  |
| 19 | 567949332395249664 | delay_or_cancellation | baggage | 0.28 | ok, I have that, pretty sure I had it before too but will wait and see what happens.... |  |
| 20 | 568185323668291584 | baggage | delay_or_cancellation | 0.42 | correct date is 2/11/15! |  |
