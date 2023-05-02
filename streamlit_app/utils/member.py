class Member:
    def __init__(
        self, name: str, linkedin_url: str = None, github_url: str = None, researchgate_url: str = None
    ) -> None:
        self.name = name
        self.linkedin_url = linkedin_url
        self.github_url = github_url
        self.researchgate_url = researchgate_url

    def sidebar_markdown(self):

        markdown = f'<span class="member">{self.name}'

        if self.linkedin_url is not None:
            markdown += f'<a href={self.linkedin_url} target="_blank"><img src="https://dst-studio-template.s3.eu-west-3.amazonaws.com/linkedin-logo-black.png" alt="linkedin" width="25"></a>'
        
        if self.researchgate_url is not None:
            markdown += f'<a href={self.researchgate_url} target="_blank"><img src="https://cdn.iconscout.com/icon/free/png-256/researchgate-3629614-3031082.png" alt="researchgate" width="19" height="18" style="margin:0 3px 0 6px"></a>'

        if self.github_url is not None:
            markdown += f'<a href={self.github_url} target="_blank"><img src="https://dst-studio-template.s3.eu-west-3.amazonaws.com/github-logo.png" alt="github" width="20"></a>'

        markdown += '</span>'

        return markdown

