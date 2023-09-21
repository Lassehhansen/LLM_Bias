gender_keywords_dict = {
"Male": [
"Male", "Man", "Men", "Boy", "Boys", "Gentleman", "Gentlemen", "Mister", "Sir", "He", "Him", "His", "Himself",
"Males", "guy", "guys", "father", "fathers", "dad", "dads", "son", "sons", "husband", "husbands", "grandfather", "grandfathers",
"grandpa", "grandpas", "brother", "brothers"
],
"Female": [
# Female
"Female", "Woman", "Women", "Girl", "Girls", "Lady", "Ladies", "Mrs", "Madam", "She", "Her", "Hers", "Females", "Herself",
"gal", "gals", "mother", "mothers", "mom", "moms", "daughter", "daughters", "wife", "wives", "grandmother", "grandmothers",
"grandma", "grandmas", "sister", "sisters", "sista", "sistas"
]
}

gender_keywords_dict = {key.lower(): [word.lower() for word in value] 
for key, value in gender_keywords_dict.items()}
