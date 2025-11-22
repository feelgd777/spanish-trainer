from openai import OpenAI

client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4.1-mini",  # or "gpt-4o-mini"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say one short sentence in Spanish."}
    ]
)

print(resp.choices[0].message.content)
