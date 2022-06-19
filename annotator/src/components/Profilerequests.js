import { useEffect, useState } from "react";
import { connect } from "react-redux";
import { getRequests, getProjectsList, sendRequest, sendRequestAction } from "../scripts/user";
import "./../styles/profile.css";

const ProfileRequests = (props) => {
    const handleErrorMessage = props.handleErrorMessage;
    const [sentRequests, setSentRequests] = useState([]);
    const [receivedRequests, setReceivedRequests] = useState([]);
    const [projectsList, setProjectsList] = useState([]);
    const [projectDisplayList, setProjectDisplayList] = useState([]);
    const [newRequestDetails, setNewRequestDetails] = useState({
        title: "",
        description: "",
        role: "0",
    });

    const handleLoadRequests = async (type) => {
        if (type === "sent") {
            let sentRequestsList = await getRequests(props.auth.token, {
                terminal: "user",
                type: "sent",
            });
            if (sentRequestsList) setSentRequests(sentRequestsList);
            else handleErrorMessage("cannot load Sent Requests, Try Again");
        } else {
            let receivedRequestsList = await getRequests(props.auth.token, {
                terminal: "user",
                type: "received",
            });
            if (receivedRequestsList) setReceivedRequests(receivedRequestsList);
            else handleErrorMessage("Cannot load Received Requests, Try Again");
        }
    };

    const handleLoadProjects = async () => {
        let allProjects = await getProjectsList(props.auth.token);
        if (allProjects) setProjectsList(allProjects);
        else handleErrorMessage("Cannot load all users data, Try Again");
    };

    useEffect(() => {
        handleLoadRequests("sent");
    }, []);

    useEffect(() => {
        setProjectDisplayList(projectsList);
    }, [projectsList]);

    const [requestProjectField, setRequestProjectField] = useState({
        _id: null,
        title: "",
        description: "",
        creator: "",
    });

    const handleRequestAction = async (id, action) => {
        let status = await sendRequestAction(props.auth.token, id, action === "accepted" ? "2" : "3");
        if (status) {
            setReceivedRequests(receivedRequests.map((request) => (request._id === id ? { ...request, status: action } : request)));
        } else handleErrorMessage("Cannot set Request Status, Try Again");
    };

    const handleRequestDetails = (e, field) => {
        if (field === "title") {
            setNewRequestDetails({
                ...newRequestDetails,
                title: e.target.value,
            });
            document.getElementById("title-invalid").style.display = "none";
        } else if (field === "description") {
            setNewRequestDetails({
                ...newRequestDetails,
                description: e.target.value,
            });
            document.getElementById("description-invalid").style.display = "none";
        } else {
            setNewRequestDetails({
                ...newRequestDetails,
                role: e.target.value,
            });
            document.getElementById("role-invalid").style.display = "none";
        }
    };

    const handleRequestProject = (e) => {
        setProjectDisplayList(
            projectsList.filter((project) => {
                if (project.title.toLowerCase().includes(e.target.value.toLowerCase()) || project.description.toLowerCase().includes(e.target.value.toLowerCase()) || project.creator.toLowerCase().includes(e.target.value.toLowerCase())) return project;
            })
        );
    };

    const handleProjectSelect = (id) => {
        setRequestProjectField(projectsList.find((project) => project._id === id));
        document.getElementById("project-invalid").style.display = "none";
    };

    const handleNewRequestSend = async () => {
        let flag = true;
        if (newRequestDetails.title === "") {
            document.getElementById("title-invalid").style.display = "inline";
            flag = false;
        }
        if (newRequestDetails.description === "") {
            document.getElementById("description-invalid").style.display = "inline";
            flag = false;
        }
        if (newRequestDetails.role === "") {
            document.getElementById("role-invalid").style.display = "inline";
            flag = false;
        }
        if (requestProjectField._id === null) {
            document.getElementById("project-invalid").style.display = "inline";
            flag = false;
        }
        if (flag) {
            let status = await sendRequest(props.auth.token, {
                title: newRequestDetails.title,
                description: newRequestDetails.description,
                role: newRequestDetails.role,
                project: requestProjectField._id,
                terminal: "user",
            });
            if (!status) {
                handleErrorMessage("Request not sent, Try Again");
            }
        }
        handleNewRequestReset();
    };

    const handleNewRequestReset = () => {
        setNewRequestDetails({
            title: "",
            description: "",
            role: "0",
        });
        setRequestProjectField({
            _id: null,
            title: "",
            description: "",
            creator: "",
        });
    };

    return (
        <div className="requests container p-3">
            <nav>
                <div className="nav nav-tabs row requests-navbar" id="nav-tab" role="tablist">
                    <button
                        className="col-4 nav-link active"
                        id="nav-sentrequests-tab"
                        data-bs-toggle="tab"
                        data-bs-target="#nav-sentrequests"
                        type="button"
                        role="tab"
                        aria-controls="nav-sentrequests"
                        aria-selected="true"
                        onClick={() => {
                            handleLoadRequests("sent");
                        }}
                    >
                        <div className="nav-button">Sent Requests</div>
                    </button>
                    <button
                        className="col-4 nav-link"
                        id="nav-receivedrequests-tab"
                        data-bs-toggle="tab"
                        data-bs-target="#nav-receivedrequests"
                        type="button"
                        role="tab"
                        aria-controls="nav-receivedrequests"
                        aria-selected="false"
                        onClick={() => {
                            handleLoadRequests("received");
                        }}
                    >
                        <div className="nav-button">Reveived Requests</div>
                    </button>
                    <button
                        className="col-4 nav-link"
                        id="nav-newrequest-tab"
                        data-bs-toggle="tab"
                        data-bs-target="#nav-newrequest"
                        type="button"
                        role="tab"
                        aria-controls="nav-newrequest"
                        aria-selected="false"
                        onClick={() => {
                            handleLoadProjects();
                        }}
                    >
                        <div className="nav-button">New Request</div>
                    </button>
                </div>
            </nav>
            <div className="tab-content" id="nav-tabContent">
                <div className="tab-pane fade show active" id="nav-sentrequests" role="tabpanel" aria-labelledby="nav-sentrequests-tab">
                    <div className="row justify-content-center request-list-container">
                        <div className="col-md-8 list-container">
                            {sentRequests.length === 0 ? (
                                <div className="parent" style={{ width: "100%", height: "150px", textAlign: "center" }}>
                                    <div className="child description">
                                        You do not have any Sent Requests. You can send a new request using the <span style={{ fontSize: "18px", fontWeight: "bold", paddingLeft: "5px", paddingRight: "5px" }}>New Request</span> Section.
                                    </div>
                                </div>
                            ) : (
                                <div className="d-grid gap-3 list p-3">
                                    {sentRequests.map((request, index) => (
                                        <div
                                            key={index}
                                            className="shadow-lg p-3 list-item rounded"
                                            style={
                                                request.status === "pending"
                                                    ? {
                                                          backgroundColor: "#4285F4",
                                                          color: "white",
                                                      }
                                                    : request.status === "accepted"
                                                    ? {
                                                          backgroundColor: "#1DB954",
                                                          color: "white",
                                                      }
                                                    : {
                                                          backgroundColor: "#ff3333",
                                                          color: "white",
                                                      }
                                            }
                                        >
                                            <div className="list-item-title">{request.title}</div>
                                            <div className="list-item-description">{request.description}</div>
                                            <div className="tag-container">
                                                <span
                                                    className="rounded tag"
                                                    style={
                                                        request.status === "pending"
                                                            ? {
                                                                  color: "#4285F4",
                                                              }
                                                            : request.status === "accepted"
                                                            ? {
                                                                  color: "#1DB954",
                                                              }
                                                            : {
                                                                  color: "#ff3333",
                                                              }
                                                    }
                                                >
                                                    {request.status}
                                                </span>
                                                <span
                                                    className="rounded tag"
                                                    style={
                                                        request.role === "owner"
                                                            ? {
                                                                  color: "#7c605c",
                                                              }
                                                            : {
                                                                  color: "#b35900",
                                                              }
                                                    }
                                                >
                                                    {request.role}
                                                </span>
                                                <span className="rounded tag" style={{ color: "#7c605c" }}>
                                                    {request.project.title}
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
                <div className="tab-pane fade" id="nav-receivedrequests" role="tabpanel" aria-labelledby="nav-receivedrequests-tab">
                    <div className="row justify-content-center request-list-container">
                        <div className="col-md-8 list-container">
                            {receivedRequests.length === 0 ? (
                                <div className="parent" style={{ width: "100%", height: "150px", textAlign: "center" }}>
                                    <div className="child description">You do not have any Received Requests</div>
                                </div>
                            ) : (
                                <div className="d-grid gap-3 list p-3">
                                    {receivedRequests.map((request, index) =>
                                        request.status === "pending" ? (
                                            <div
                                                key={index}
                                                className="shadow-lg p-3 list-item rounded"
                                                style={
                                                    request.status === "pending"
                                                        ? {
                                                              backgroundColor: "#4285F4",
                                                              color: "white",
                                                          }
                                                        : request.status === "accepted"
                                                        ? {
                                                              backgroundColor: "#1DB954",
                                                              color: "white",
                                                          }
                                                        : {
                                                              backgroundColor: "#ff3333",
                                                              color: "white",
                                                          }
                                                }
                                            >
                                                <div className="row">
                                                    <div className="col-10">
                                                        <div className="list-item-title">{request.title}</div>
                                                        <div className="list-item-description">{request.description}</div>
                                                        <div className="tag-container">
                                                            <span
                                                                className="rounded tag"
                                                                style={
                                                                    request.status === "pending"
                                                                        ? {
                                                                              color: "#4285F4",
                                                                          }
                                                                        : request.status === "accepted"
                                                                        ? {
                                                                              color: "#1DB954",
                                                                          }
                                                                        : {
                                                                              color: "#ff3333",
                                                                          }
                                                                }
                                                            >
                                                                {request.status}
                                                            </span>
                                                            <span
                                                                className="rounded tag"
                                                                style={
                                                                    request.role === "owner"
                                                                        ? {
                                                                              color: "#7c605c",
                                                                          }
                                                                        : {
                                                                              color: "#b35900",
                                                                          }
                                                                }
                                                            >
                                                                {request.role}
                                                            </span>
                                                            <span className="rounded tag" style={{ color: "#7c605c" }}>
                                                                {request.project.title}
                                                            </span>
                                                        </div>
                                                    </div>
                                                    <div
                                                        className="col-2 delete-icon-container"
                                                        style={{
                                                            padding: "0px",
                                                        }}
                                                    >
                                                        <div
                                                            className="delete-icon"
                                                            style={{
                                                                textAlign: "center",
                                                                marginTop: "auto",
                                                            }}
                                                        >
                                                            <div
                                                                className="shadow rounded accept-button p-2"
                                                                onClick={() => {
                                                                    handleRequestAction(request._id, "accepted");
                                                                }}
                                                            >
                                                                <i className="fas fa-check fa-lg"></i>
                                                            </div>
                                                            <div
                                                                className="shadow rounded decline-button p-2"
                                                                onClick={() => {
                                                                    handleRequestAction(request._id, "declined");
                                                                }}
                                                            >
                                                                <i className="fas fa-times fa-lg"></i>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        ) : (
                                            <div
                                                key={index}
                                                className="shadow-lg p-3 list-item rounded"
                                                style={
                                                    request.status === "accepted"
                                                        ? {
                                                              backgroundColor: "#1DB954",
                                                              color: "white",
                                                          }
                                                        : {
                                                              backgroundColor: "#ff3333",
                                                              color: "white",
                                                          }
                                                }
                                            >
                                                <div className="list-item-title">{request.title}</div>
                                                <div className="list-item-description">{request.description}</div>
                                                <div className="tag-container">
                                                    <span
                                                        className="rounded tag"
                                                        style={
                                                            request.status === "pending"
                                                                ? {
                                                                      color: "#4285F4",
                                                                  }
                                                                : request.status === "accepted"
                                                                ? {
                                                                      color: "#1DB954",
                                                                  }
                                                                : {
                                                                      color: "#ff3333",
                                                                  }
                                                        }
                                                    >
                                                        {request.status}
                                                    </span>
                                                    <span
                                                        className="rounded tag"
                                                        style={
                                                            request.role === "owner"
                                                                ? {
                                                                      color: "#7c605c",
                                                                  }
                                                                : {
                                                                      color: "#b35900",
                                                                  }
                                                        }
                                                    >
                                                        {request.role}
                                                    </span>
                                                    <span className="rounded tag" style={{ color: "#7c605c" }}>
                                                        {request.project.title}
                                                    </span>
                                                </div>
                                            </div>
                                        )
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
                <div className="tab-pane fade" id="nav-newrequest" role="tabpanel" aria-labelledby="nav-newrequest-tab">
                    <div className="row request-list-container justify-content-center" style={{ marginTop: "20px" }}>
                        <div className="col-md-7">
                            <div className="row g-3">
                                <div className="col-md-12">
                                    <label htmlFor="title" className="form-label">
                                        Title
                                    </label>
                                    <input
                                        type="text"
                                        className="form-control"
                                        id="title"
                                        value={newRequestDetails.title}
                                        onChange={(e) => {
                                            handleRequestDetails(e, "title");
                                        }}
                                    />
                                    <span id="title-invalid" className="field-invalid">
                                        field is required
                                    </span>
                                </div>
                                <div className="col-md-8">
                                    <label htmlFor="description" className="form-label">
                                        Description
                                    </label>
                                    <textarea
                                        className="form-control"
                                        id="description"
                                        value={newRequestDetails.description}
                                        style={{ height: "140px" }}
                                        onChange={(e) => {
                                            handleRequestDetails(e, "description");
                                        }}
                                    ></textarea>
                                    <span id="description-invalid" className="field-invalid">
                                        field is required
                                    </span>
                                </div>
                                <div className="col-md-4">
                                    <div className="d-grid gap-3">
                                        <div>
                                            <label htmlFor="type" className="form-label">
                                                Role
                                            </label>
                                            <select
                                                className="form-control"
                                                id="type"
                                                defaultValue=""
                                                onChange={(e) => {
                                                    handleRequestDetails(e, "role");
                                                }}
                                            >
                                                <option disabled value="">
                                                    Choose...
                                                </option>
                                                <option value="1">owner</option>
                                                <option value="2">staff</option>
                                            </select>
                                            <span id="role-invalid" className="field-invalid">
                                                field is required
                                            </span>
                                        </div>
                                        <div>
                                            <label htmlFor="username" className="form-label">
                                                Project
                                            </label>
                                            <input type="text" className="form-control" id="request-username" value={requestProjectField.title} disabled aria-describedby="request-username-help" />
                                            <div id="request-username-help" className="form-text">
                                                Select using the projects section
                                            </div>
                                            <span id="project-invalid" className="field-invalid">
                                                field is required
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="col-md-4">
                            <div className="container">
                                <div className="row justify-content-start search-bar-container">
                                    <div className="col-2 col-sm-2 col-md-1 search-icon">
                                        <i className="fas fa-search fa-lg"></i>
                                    </div>
                                    <div className="col-10 col-sm-10 col-md-11">
                                        <div className="row">
                                            <input
                                                type="text"
                                                className="search-bar"
                                                placeholder="Search Project"
                                                onChange={(e) => {
                                                    handleRequestProject(e);
                                                }}
                                            />
                                        </div>
                                    </div>
                                </div>
                                <div className="row border border-top-0 border-dark shadow-lg">
                                    <div className="search-box-container">
                                        {projectDisplayList.map((project, index) => (
                                            <div
                                                key={index}
                                                className="search-box-item border rounded shadow-sm"
                                                onClick={() => {
                                                    handleProjectSelect(project._id);
                                                }}
                                            >
                                                <div className="name">{project.title}</div>
                                                <div className="description">{project.description}</div>
                                                <div className="username-container">
                                                    <span className="username-helpertext">created by:</span>
                                                    <span className="username">{project.creator}</span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="container p-3">
                            <button
                                className="btn btn-primary request-send-button"
                                onClick={() => {
                                    handleNewRequestSend();
                                }}
                            >
                                Send
                            </button>
                            <button
                                className="btn btn-danger request-reset-button"
                                onClick={() => {
                                    handleNewRequestReset();
                                }}
                            >
                                Reset
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default connect((state) => ({ auth: state.auth }))(ProfileRequests);
