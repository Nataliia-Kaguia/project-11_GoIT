from django import forms


class ChurnPredictionForm(forms.Form):
    is_tv_subscriber = forms.ChoiceField(choices=[(0, "No"), (1, "Yes")])
    is_movie_package_subscriber = forms.ChoiceField(choices=[(0, "No"), (1, "Yes")])

    subscription_age = forms.IntegerField(min_value=0)
    bill_avg = forms.FloatField(min_value=0)
    reamining_contract = forms.IntegerField(min_value=0)
    service_failure_count = forms.IntegerField(min_value=0)

    download_avg = forms.FloatField(min_value=0)
    upload_avg = forms.FloatField(min_value=0)

    download_over_limit = forms.ChoiceField(choices=[(0, "No"), (1, "Yes")])
