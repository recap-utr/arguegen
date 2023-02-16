import openai

openai.api_key_path = "./openai_api_key.txt"
QUERY = "Should there be a cap on annuity increases for a change of remunerators?"
CASE = (
    "Rent prices should be limited by a cap when there's a change of tenant. Landlords"
    " may want to earn as much as possible, and many, consistent with market"
    " principles, are prepared to pay higher rents, but that people with the same"
    " income suddenly must pay more and can't live in the same flat anymore seems"
    " implausible. Gentrification destroys entire districts and their culture."
)


res = openai.Edit.create(
    model="text-davinci-edit-001",
    input=CASE,
    instruction=f"Make it more relevant to {QUERY}",
)

print(res.choices[0].text)  # type: ignore


# EXAMPLE OUTPUT:
#
# instruction=f"Make it more relevant to {QUERY}", temperature=None, top_p=None
# Annuity prices should be limited by a cap when there's a change of remunerator. Remunerators may want to earn as much as possible, but that people with the same income suddenly must pay more and can't work for the same remunerator anymore seems implausible. Banks demand appraisals, never accept deposits at appraised value, and may impose arbitrary limits.
#
# instruction=f"Adapt topic from rent increases to annuity increases", temperature=None, top_p=None
# Annuities taken out by pensioners prevent many families from being threatened by poverty in old age. For the elderly these annuities are higher when they are first taken out, as it is easier to predict mortality then, than it is in later stages of your life. The annuity markets should therefore allow for the annuity to change with one's advancing age.
