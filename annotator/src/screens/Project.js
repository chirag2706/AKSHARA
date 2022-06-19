import { useEffect, useState } from "react";
import { connect } from "react-redux";
import { useHistory } from "react-router-dom";
import { useParams } from "react-router-dom";
import Createprojectmodal from "../components/modals/Createprojectmodal";
import Uploaddocumentmodal from "../components/modals/Uploaddocumentmodal";
import Projectinfo from "../components/project/Projectinfo";
import Projectrequests from "../components/project/Projectrequests";
import UserList from "../components/project/Userlist";
import Sidebar from "../components/Sidebar";
import { createProject, deleteUser, getProjectUsers, getUserProjects, leaveProject } from "../scripts/project";
import "../styles/project.css";

const Project = (props) => {
    let history = useHistory();

    if (!props.auth.token) {
        history.push("/login");
    }

    const { id } = useParams();
    const [loading, setLoading] = useState(true);

    const [ownedProjects, setOwnedProjects] = useState(null);
    const [sharedProjects, setSharedProjects] = useState(null);
    const [isOwner, setIsOwner] = useState(false);
    const [validProject, setValidProject] = useState(true);

    const [owners, setOwners] = useState([]);
    const [staff, setStaff] = useState([]);

    const [creatingProject, setCreatingProject] = useState(false);
    const handleCreateProject = async () => {
        let title = document.getElementById("createModalTitleInput").value;
        let description = document.getElementById("createModalDescriptionInput").value;
        if (title.split(" ").join("") === "") {
            document.getElementById("createModalTitleHelperText").style.display = "inline";
        } else if (description.split(" ").join("") === "") {
            document.getElementById("createModalDescriptionHelperText").style.display = "inline";
        } else {
            document.getElementById("createModalTitleHelperText").style.display = "none";
            document.getElementById("createModalDescriptionHelperText").style.display = "none";
            setCreatingProject(true);
            let status = await createProject(props.auth.token, { title, description });
            if (status) {
                setCreatingProject(false);
                document.getElementById("createProjectModalClose").click();
                window.location.reload()
            }
        }
    };

    const [projectInfo, setProjectInfo] = useState({
        _id: null,
        title: "Project 1",
        description: "This is Project 1",
        creator: "user1",
    });

    const loadUsers = async (id) => {
        let users = await getProjectUsers(props.auth.token, id);
        setOwners(users["owners"]);
        setStaff(users["staff"]);
    };

    const handleDelete = async (username, type) => {
        let status = await deleteUser(props.auth.token, username, projectInfo._id);
        if (status) {
            if (type === "owner") {
                setOwners(owners.filter((owner) => owner.username !== username));
            } else {
                setStaff(staff.filter((owner) => owner.username !== username));
            }
        }
    };

    const handleLeaveProject = async () => {
        let status = await leaveProject(props.auth.token, projectInfo._id);
        if (status) {
            setOwnedProjects(ownedProjects.filter((project) => project._id !== projectInfo._id))
            setSharedProjects(sharedProjects.filter((project) => project._id !== projectInfo._id))
            history.push("/projects");
        }
    };

    useEffect(() => {
        setLoading(true);
        async function loadProjects() {
            let projectsList = await getUserProjects(props.auth.token);
            if (projectsList) {
                setOwnedProjects(projectsList["owned_projects"]);
                setSharedProjects(projectsList["shared_projects"]);
            }
        }

        loadProjects();
    }, []);

    useEffect(() => {
        if(sharedProjects !== null) {
            if (id) {
                let currentProject = ownedProjects.find((project) => project._id === id);
                if (currentProject) {
                    setIsOwner(true);
                    setProjectInfo(currentProject);
                    loadUsers(id);
                    setValidProject(true);
                    setLoading(false);
                } else {
                    currentProject = sharedProjects.find((project) => project._id === id);
                    if (currentProject) {
                        setIsOwner(false);
                        setProjectInfo(currentProject);
                        loadUsers(id);
                        setValidProject(true);
                        setLoading(false);
                    } else {
                        setLoading(false);
                        setValidProject(false);
                    }
                }
            } else {
                if (ownedProjects.length > 0) {
                    history.push("/projects/" + ownedProjects[0]._id);
                } else if (sharedProjects.length > 0) {
                    history.push("/projects/" + sharedProjects[0]._id);
                } else {
                    setProjectInfo({
                        _id: null,
                        title: "",
                        description: "",
                        creator: "",
                    });
                    setValidProject(true);
                    setLoading(false);
                }
            }
        }
    }, [sharedProjects, id]);

    return (
        <div style={{ height: "100%" }}>
            {loading ? (
                <div style={{ paddingTop: "100px" }}>
                    <div className="d-flex justify-content-center">
                        <div className="spinner-border" role="status" style={{ color: "#4285F4" }}></div>
                    </div>
                    <div style={{ textAlign: "center", paddingTop: "20px" }}>Loading projects. Please wait</div>
                </div>
            ) : !validProject ? (
                <div style={{ paddingTop: "100px" }}>
                    <div className="d-flex justify-content-center">
                        <i className="fa-solid fa-bomb fa-xl"></i>
                    </div>
                    <div style={{ textAlign: "center", paddingTop: "20px" }}>Sorry. Project cannot be found.</div>
                </div>
            ) : projectInfo._id ? (
                <div>
                    <Sidebar ownedProjects={ownedProjects} sharedProjects={sharedProjects} project_id={projectInfo._id} type="2" path="/projects" />
                    <Createprojectmodal creatingProject={creatingProject} handleCreateProject={handleCreateProject} />
                    <Projectinfo projectInfo={projectInfo} setProjectInfo={setProjectInfo} handleLeaveProject={handleLeaveProject} />
                    <div className="list-main-container">
                        <div className="container">
                            <div className="row justify-content-evenly">
                                <UserList title="Owners" list={owners} isOwner={isOwner} type="owner" creator={projectInfo.creator} handleDelete={handleDelete} />
                                <UserList title="Staff" list={staff} isOwner={isOwner} type="staff" creator={projectInfo.creator} handleDelete={handleDelete} />
                            </div>
                        </div>
                    </div>
                    {isOwner ? <Projectrequests project_id={projectInfo._id} /> : <></>}
                </div>
            ) : (
                <div>
                    <Sidebar ownedProjects={ownedProjects} sharedProjects={sharedProjects} project_id={projectInfo._id} type="2" path="/projects" />
                    <Createprojectmodal creatingProject={creatingProject} handleCreateProject={handleCreateProject} />
                    <div style={{ paddingTop: "100px" }}>
                        <div className="d-flex justify-content-center">
                            <i className="fa-solid fa-face-frown fa-xl"></i>
                        </div>
                        <div style={{ textAlign: "center", paddingTop: "20px" }}>No Projects found.</div>
                    </div>
                </div>
            )}
        </div>
    );
};

// export default Project;
export default connect((state) => ({ auth: state.auth }))(Project);
