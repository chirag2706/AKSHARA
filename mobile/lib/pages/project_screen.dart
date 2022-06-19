import 'dart:convert';
import 'package:akshara/pages/login_screen.dart';
import 'package:akshara/pages/upload_document.dart';
import 'package:akshara/providers/auth_provider.dart';
import 'package:akshara/utils/constants.dart';
import 'package:akshara/utils/user_utils.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';

class Projects extends StatefulWidget {
  const Projects({Key? key}) : super(key: key);

  @override
  State<Projects> createState() => _ProjectsState();
}

class _ProjectsState extends State<Projects> {
  var ownedprojects = [];
  var sharedprojects = [];

  @override
  void initState() {
    super.initState();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    loadProjects();
  }

  void loadProjects() async {
    final response = await http
        .get(Uri.parse(AppConstants.baseUrl + '/get-user-projects'), headers: {
      "authtoken": Provider.of<AuthProvider>(context).user.token.toString()
    });
    setState(() {
      var body = jsonDecode(response.body);
      ownedprojects = body['owned_projects'];
      sharedprojects = body['shared_projects'];
    });
  }

  Widget itemBuilder(BuildContext context, int index, String type) {
    return GestureDetector(
      onTap: () {
        Navigator.of(context).push(MaterialPageRoute(
            builder: (context) => UploadDocument(
                projectId: (type == 'owned')
                    ? ownedprojects[index]['_id']
                    : sharedprojects[index]['_id'])));
      },
      child: Card(
          clipBehavior: Clip.antiAlias,
          elevation: 5,
          child: (type == 'owned')
              ? ListTile(
                  title: Text(ownedprojects[index]['title']),
                  subtitle: Text(ownedprojects[index]['creator']),
                )
              : ListTile(
                  title: Text(sharedprojects[index]['title']),
                  subtitle: Text(sharedprojects[index]['creator']),
                )),
    );
  }

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
        length: 2,
        child: Scaffold(
          appBar: AppBar(
            bottom: const TabBar(
              tabs: [
                Tab(
                  text: "Owned",
                ),
                Tab(
                  text: "Shared",
                ),
              ],
            ),
            title: const Text('Projects'),
            actions: [
              IconButton(
                  onPressed: () {
                    UserUtils.logOut();
                    Navigator.pop(context);
                    Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (context) => const LoginPage()));
                  },
                  icon: const Icon(Icons.logout))
            ],
          ),
          body: TabBarView(
            children: [
              ListView.builder(
                  itemCount: ownedprojects.length,
                  itemBuilder: ((context, index) =>
                      itemBuilder(context, index, "owned"))),
              ListView.builder(
                  itemCount: sharedprojects.length,
                  itemBuilder: ((context, index) =>
                      itemBuilder(context, index, "shared")))
            ],
          ),
        ));
  }
}
