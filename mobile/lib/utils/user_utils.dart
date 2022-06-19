import 'dart:convert';
import 'package:akshara/utils/constants.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;
import '../models/user.dart';
import 'dart:async';

class UserUtils {
  static Future<String> login(data) async {
    final response = await http.post(Uri.parse(AppConstants.baseUrl + "/login"),
        body: {"username": data['username'], "password": data['password']});
    if (response.statusCode == 200) {
      await saveUser(User(token: jsonDecode(response.body)['jwt']));
      return jsonDecode(response.body)['jwt'];
    }
    return '';
  }

  static Future<String> register(data) async {
    final response = await http
        .post(Uri.parse(AppConstants.baseUrl + "/register"), body: {
      "name": data['name'],
      "username": data['username'],
      "password": data['password']
    });
    if (response.statusCode == 200) {
      await saveUser(User(token: jsonDecode(response.body)['jwt']));
      return jsonDecode(response.body)['jwt'];
    }
    return '';
  }

  static Future<bool> saveUser(User user) async {
    final pref = await SharedPreferences.getInstance();
    if (user.token != null) {
      return await pref.setString("token", user.token.toString());
    }
    return false;
  }

  static Future<User> getUser() async {
    final pref = await SharedPreferences.getInstance();
    final token = pref.getString("token");
    return User(token: token);
  }

  static Future<bool> isLoggedIn() async {
    final pref = await SharedPreferences.getInstance();
    final token = pref.getString("token");
    if (token == null) return false;

    final response = await http.post(
        Uri.parse(AppConstants.baseUrl + "/validate-token"),
        headers: {'authtoken': token.toString()});

    if (response.statusCode == 200) {
      return true;
    }
    return false;
  }

  static Future<bool> logOut() async {
    final pref = await SharedPreferences.getInstance();
    return pref.remove("token");
  }
}
