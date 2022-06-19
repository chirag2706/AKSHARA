import 'package:akshara/models/user.dart';
import 'package:flutter/material.dart';

class AuthProvider extends ChangeNotifier {
  User _user = User();
  User get user => _user;

  void setUser(User user) {
    _user = user;
    notifyListeners();
  }
}
