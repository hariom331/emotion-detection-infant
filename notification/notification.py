from pushbullet import Pushbullet

api_key = 'o.KXBun4UI1n9BG3AMIdR3iD2ik8f9dQMq'
pb = Pushbullet(api_key)



# Replace TITLE and MESSAGE with your desired notification title and message
push = pb.push_note("TITLE", "MESSAGE")

