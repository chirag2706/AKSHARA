class User {
  String? token;

  User({this.token});

  factory User.fromJson(response) {
    return User(token: response['jwt']);
  }
}
