import 'package:akshara/pages/login_screen.dart';
import 'package:akshara/pages/project_screen.dart';
import 'package:akshara/providers/auth_provider.dart';
import 'package:akshara/utils/user_utils.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    Future<bool> checkLoggedIn() => UserUtils.isLoggedIn();

    return MultiProvider(
      providers: [ChangeNotifierProvider(create: (_) => AuthProvider())],
      child: MaterialApp(
        title: 'Akshara',
        theme: ThemeData(
          primarySwatch: Colors.blue,
          visualDensity: VisualDensity.adaptivePlatformDensity,
        ),
        home: FutureBuilder(
            future: checkLoggedIn(),
            builder: (context, snapshot) {
              if (snapshot.hasData) {
                if (snapshot.data == true) {
                  return const Projects();
                } else {
                  return const LoginPage();
                }
              }
              return const Scaffold(
                  body: Center(
                      child: Text(
                "Akshara",
                style: TextStyle(fontSize: 40, fontWeight: FontWeight.bold),
              )));
            }),
      ),
    );
  }
}
