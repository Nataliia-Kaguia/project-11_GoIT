from django import forms


class ChurnPredictionForm(forms.Form):
    subscription_age = forms.FloatField(label="Subscription Age (months)")
    bill_avg = forms.FloatField(label="Average Bill ($)")
    reamining_contract = forms.FloatField(label="Remaining Contract (months)")
    service_failure_count = forms.FloatField(label="Service Failure Count")
    download_avg = forms.FloatField(label="Download Avg (GB)")
    upload_avg = forms.FloatField(label="Upload Avg (GB)")

    is_tv_subscriber = forms.ChoiceField(
        choices=[(0, "No"), (1, "Yes")], widget=forms.Select
    )
    is_movie_package_subscriber = forms.ChoiceField(
        choices=[(0, "No"), (1, "Yes")], widget=forms.Select
    )
    download_over_limit = forms.ChoiceField(
        choices=[(0, "No"), (1, "Yes")], widget=forms.Select
    )
