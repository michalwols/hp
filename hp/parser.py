def camel_to_snake(text):
  s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def abbreviate(text):
  return re.sub(r"([a-zA-Z])[a-z]*[^A-Za-z]*",r"\1", text).lower()


def get_arg_parser(x, description=None, epilog=None, parser=None, **kwargs):
  import argparse
  from ..params import Field
  parser = parser or argparse.ArgumentParser(description=description, epilog=epilog, **kwargs)
  for k, v in x.items():
    if isinstance(v, dict):
      parser.add_argument(
        f"-{abbreviate(k)}",
        f"--{camel_to_snake(k)}",
        default=v.get('default'),
        type=v.get('type'),
        action=v.get('action'),
        help=v.get('help'),
        required=v.get('required'),
        choices=v.get('choices'),
        dest=v.get('dest')
      )
    elif isinstance(v, Field):
      parser.add_argument(
        f"-{abbreviate(k)}",
        f"--{camel_to_snake(k)}",
        default=v.default,
        type=v.type,
        help=f"{v.help or k} (default: {v.default})",
        required=v.required,
        choices=getattr(v, 'choices', None),
      )
    else:
      parser.add_argument(f"-{abbreviate(k)}", f"--{camel_to_snake(k)}", default=v, type=type(v))
  return parser